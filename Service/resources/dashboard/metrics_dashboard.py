import asyncio
import json
import os
import time
from time import perf_counter


import matplotlib.pyplot as plt
import requests
import pandas as pd
import streamlit as st
from tinydb import Query, TinyDB

import client_collector
import utils
from dot_pipeline_gen import get_dot_graph
from k8s import restart_application


st.set_page_config(
    page_title="Metrics",
    layout="wide",
    initial_sidebar_state="expanded",
)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# * Setting address for resource collection server
# RESOURCE_COLLECTOR = os.getenv("RESOURCE_COLLECTOR")
# if not RESOURCE_COLLECTOR:
#     RESOURCE_COLLECTOR = "http://143.248.53.10:30000"

# client_collector.RESOURCE_COLLECTOR = RESOURCE_COLLECTOR


@st.cache(allow_output_mutation=True)
def persist_data():
    return [], {}


db = TinyDB("pipelines.json")
Pipelines = Query()

device_urls, devices = persist_data()

global_queues = {}
global_metrics = {}


def update_pipeline(deploy=True):

    components = db.search(Pipelines.name == deploy_choice)[0]
    device_urls.clear()
    devices.clear()

    info_prog.progress(0)
    cleanup_cnt = 0
    deploy_cnt = 0
    total = len(components["pipeline"])
    total_prog = total * 2 if deploy else total
    starttime = perf_counter()
    for pipeline_part in components["pipeline"]:
        host = pipeline_part["device_url"]
        device_urls.append(host)
        body = pipeline_part["request"]
        info_text.write("Cleaning up {}/{} ..".format(cleanup_cnt+1, total))
        # * Before we deploy we want to make sure that previous services are stopped and deleted
        cleanup = requests.delete("http://{}/stop".format(host))
        if cleanup.status_code != 200:
            st.error("Could not cleanup components on " + host)
        cleanup_cnt += 1
        info_prog.progress(
            int(((cleanup_cnt + deploy_cnt) / total_prog) * 100))

        if deploy:
            info_text.write("Deploying {}/{} ..".format(deploy_cnt+1, total))
            response = requests.post("http://{}/link".format(host), json=body)
            if response.status_code != 200:
                st.error("Could not deploy components to " + host)
            deploy_cnt += 1
            info_prog.progress(
                int(((cleanup_cnt + deploy_cnt) / total_prog) * 100))
    info_text.write("")
    info_prog.write("")
    print("took {} seconds".format(perf_counter() - starttime))



plt.style.use("ggplot")

# * Adding a pipeline to database
pipeline_json = st.sidebar.file_uploader(
    "Add new pipeline from json file", type=["json"])
if pipeline_json is not None:
    pipeline = json.load(pipeline_json)["requests"]
    filename = pipeline_json.name.replace(".json", "")

    if db.search(Pipelines.name == filename):
        st.sidebar.info("Pipeline already exists")
    else:
        db.insert({"name": filename, "pipeline": pipeline})

# * Deploying a pipeline from database
deploy_choice = st.sidebar.selectbox("Select pipeline to deploy", [
                                     pipeline["name"] for pipeline in db.all()])
with st.sidebar.expander("See deployment information"):
    try:
        st.graphviz_chart(get_dot_graph(
            db.search(Pipelines.name == deploy_choice)[0]["pipeline"]))
    except IndexError:
        pass

# * Controls for deploying and deleting pipelines
btn_col, info_col = st.sidebar.columns(2)
info_text = info_col.empty()
info_prog = info_col.empty()

if btn_col.button("Deploy pipeline"):
    update_pipeline(deploy=True)

if btn_col.button("Delete/Stop pipeline"):
    update_pipeline(deploy=False) 

# * Deleting pipelines from database
delete_choice = st.sidebar.multiselect("Select pipeline to delete from DB", [
                                       pipeline["name"] for pipeline in db.all()])
if st.sidebar.button("Delete selected pipelines"):
    for to_delete in delete_choice:
        db.remove(Pipelines.name == to_delete)

# * Accordion pane for restarting k8s app
with st.sidebar.expander("Restart k8s application"):
    name = st.text_input("Deployment name", value="ip-cameras")
    namespace = st.text_input("Namespace name", value="testbed-cameras")

    type = st.radio("Deployment type", ["daemonset", "deployment"])
    restart = st.button("Restart")
    get_status = st.button("Get Status")
    restart_progress = st.progress(0)
    restart_status = st.empty()

    if restart:
        restart_application(type, name, namespace,
                            restart_progress, restart_status)
    elif get_status:
        status = restart_application(
            type, name, namespace, restart_progress, restart_status, restart=False)
        if status:
            restart_status.write("Restart is complete")

# * Sliders for quantile and duration values
utils.add_quantile()
duration = st.sidebar.slider(
    label="Experiment duration (s)", min_value=5, max_value=120, value=5)

utils.change_btn_color(True)

# * Controls for starting and monitoring execution
resource_col, monitor_col = st.sidebar.columns(2)
resource = resource_col.checkbox("Resource collection")
monitor = monitor_col.checkbox("Monitoring")
# resource_col = monitor_col.checkbox("Resource collection")
start = st.sidebar.button("Start pipeline")
# * Information widgets for execution progress
exec_progress = st.sidebar.empty()
# pipeline_bar = st.sidebar.progress(0)

download_results = st.sidebar.checkbox("Download results")


# * Get number of necessary columns and create them. Updated when we deploy
st_columns, st_rows = utils.choose_subplot_dimensions(len(device_urls))
if st_columns:
    cols = st.columns(st_columns)
    for i, device_url in enumerate(device_urls):
        dev_column = cols[i % st_columns]
        dev_column.subheader("Device {} @ {}".format(i, device_url))
        plt_metrics = dev_column.empty()
        plt_queues = dev_column.empty() 
        devices[i] = {
            "address": device_url,
            "plt_metrics": plt_metrics,
            "plt_queues": plt_queues
        }
        global_queues[device_url] = {}

# * Main pipeline execution logic
if start:
    exec_progress.write("Progress: {}".format("..."))
    utils.change_btn_color(False)
    elapsed = 0

    # * Starts resource recording
    # TODO uncomment for resources
    # if resource:
    #     exec_progress.write("Progress: {}".format("set devices"))
    #     client_collector.send_deviceupdate()
    #     exec_progress.write("Progress: {}".format("start resource collection"))
    #     client_collector.send_start()

    # * Starts pipeline
    exec_progress.write("Progress: {}".format("starting pipeline"))
    components = db.search(Pipelines.name == deploy_choice)[0]
    # head = db.search(Pipelines.name == deploy_choice)[0]["pipeline"][0]
    # host = head["device_url"]
    # TODO Check if has decoder, if yes then start
    heads_urls = [dev["device_url"] for dev in components["pipeline"] if any("Decoder" in key for key in dev["request"].keys())]
    for host in heads_urls:
        response = requests.post("http://{}/start".format(host))
        if response.status_code != 200:
            st.error("Could not start pipeline at " + host)

    # * Runs iterations as set with duration slider
    exec_progress.write("Progress: {}".format("running"))
    time.sleep(2)
    # elapsed = duration - 1
    queues_running = True
    while elapsed < duration:# and queues_running:
    # while queues_running:
        time.sleep(1)
        if not start:
            break

        if monitor:
            loop = asyncio.get_event_loop()
            queues_running = loop.run_until_complete(utils.plot_all(
                devices, global_metrics, global_queues))

        elapsed += 1
        # pipeline_bar.progress(int(elapsed / duration * 100))

    exec_progress.write("Progress: {}".format("stopping pipeline"))
    update_pipeline(deploy=False)  # * Stop and delete components

    # if resource:
    #     # * Stops resource collection
    #     exec_progress.write("Progress: {}".format("stopping resource collection"))
    #     client_collector.send_stop()
    #     # * Downloads resource consumption results
    #     exec_progress.write("Progress: {}".format("downloading resource results"))
    #     client_collector.send_download("{}_consumption".format(deploy_choice))

    utils.change_btn_color(True)
    exec_progress.write("Progress: {}".format("Complete"))
    # * Downloading results from streamlit in xlsx form
    if download_results:
        with open("{}_metrics.json".format(deploy_choice), "w") as file:
            json.dump(global_metrics, file)
        print("file downloaded")
    pipeline_bar.progress(0)


