import asyncio
import math

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

quantile = None


def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/3), 3


def add_quantile():
    global quantile
    quantile = st.sidebar.slider(label="Slider for quantile filtering",
                                 min_value=0.0, max_value=1.0, value=0.95)


def rec(values):
    vals = np.array(values)
    filtered = vals
    try:
        filtered = vals[vals < np.quantile(vals, quantile)]
    except:
        pass

    return filtered


async def plot_all(devices, global_metrics, global_queues):
    # * Prepare calls for collecting metrics
    metrics = [get_metrics(global_metrics, **dev) for dev in devices.values()]
    results_metrics = await asyncio.gather(*metrics)

    # * Prepare calls for collecting queues
    queues = [get_queues(global_queues, **dev) for dev in devices.values()]
    results_queues = await asyncio.gather(*queues)

    # * Prepare plots asynchronously
    plots = [plot(metric, queue, chart, plt_queues) for (metric, chart),
             (queue, plt_queues) in zip(results_metrics, results_queues)]
    results_plots = await asyncio.gather(*plots)

    # * Write plots to dashboard
    for fig_plt, plt_metrics, fig_q, plt_queues in results_plots:
        plt_metrics.write(fig_plt)
        plt_queues.write(fig_q)
        plt.close(fig_plt)
        plt.close(fig_q)

    queues_running = False
    for queue, _ in results_queues:
        for service, qs in queue.items():
            for name, vals in qs.items():
                if vals[-5:] != [0] * 5: # Check if last 5 calls were empty
                    return True
                
    return queues_running#, [metric for metric, plot in results_metrics] # If it returns False, then we can stop the recording

async def get_metrics(global_metrics, address, plt_metrics, plt_queues):
    """[summary]
    Function to pull new metrics from a device
    And then aggregate them with existing ones 

    Args:
        global_metrics (dict): Global metrics dict
        address (str): Device IP address
        chart (st.empty): Streamlit placeholder element to plot chart in

    Returns:
        [type]: [description]
    """

    data = requests.get("http://{}/metrics".format(address)).json()
    if address not in global_metrics:
        global_metrics[address] = data

    for service, metrics in data.items():
        # if service in ["Postprocess", "Decoder"]:
        #     continue
        for metric, values in metrics.items():
            # if "remote" in metric or "network" in metric:
            #     continue
            # if "Reid" == service and "process" == metric:
            #     continue
            # if "Inference" == service and "process" == metric:
            #     continue
            if values["list"]:
                # if metric == "pipeline":
                #     print(values["list"])
                global_metrics[address][service][metric]["list"].extend(
                    values["list"])

    return global_metrics[address], plt_metrics


async def get_queues(global_queues, address, plt_metrics, plt_queues):
    """[summary]
    Function to pull new metrics from a device
    And then aggregate them with existing ones 

    Args:
        global_metrics (dict): Global metrics dict
        address (str): Device IP address
        chart (st.empty): Streamlit placeholder element to plot chart in (Not used here)

    Returns:
        dict: Preprocessed dictionary to use for plotting
    """
    data = requests.get("http://{}/queue_capacity".format(address)).json()
    for service, qs in data.items():
        for name, vals in qs.items():
            if vals["size"] > 20:
                print(service, name, vals["size"])
            if service in global_queues[address]:
                global_queues[address][service].append(vals["size"])
            else:
                global_queues[address][service] = [vals["size"]]
            data[service][name] = global_queues[address][service]

    return data, plt_queues


async def plot(data, queues, plt_metrics, plt_queues):
    lines = {}

    try:
        for service, metrics in data.items():
            for metric, values in metrics.items():
                filtered = rec(values["list"][-100:])
                if filtered.any():
                    label = "{}_{}".format(service, metric)
                    # print("AVG for {} is {}".format(label, sum(values["list"]) / len(values["list"])))
                    if "prepare" in label:
                        print(values["skipped"], values["prepared"])
                    # if "Postprocess" in label or "Decoder" in label:
                    if "Decoder" in label:
                        # print(label, "Postprocess" in label, "Decoder" in label)
                        pass
                    if "remote" in label or "network" in label:
                        # print(label, "remote" in label, "network" in label)
                        continue
                    # if "Reid" in label and "process" in label:
                    #     continue
                    if "Inference" in label and "process" in label:
                        continue
                    lines[label] = filtered

        # * +1 is for the queue capacities
        # TODO Can we avoid calling subplots all the time?
        dimensions = choose_subplot_dimensions(len(lines))
        fig_plt = None
        if dimensions[0]:
            fig_plt, axs = plt.subplots(nrows=dimensions[0], ncols=dimensions[1])

            if dimensions[1] > 1:
                plt_cnt_x, plt_cnt_y = 0, 0
                for label, line in lines.items(): 
                    axs[plt_cnt_x, plt_cnt_y].plot(line, label=label)
                    axs[plt_cnt_x, plt_cnt_y].legend()
                    plt_cnt_y += 1
                    if plt_cnt_y == dimensions[1]:
                        plt_cnt_y = 0
                        plt_cnt_x += 1
            else:
                plt_cnt_y = 0
                if dimensions[0] == 1:
                    axs = [axs]
                for label, line in lines.items(): 
                    axs[plt_cnt_y].plot(line, label=label)
                    axs[plt_cnt_y].legend()
                    plt_cnt_y += 1


        queue_ax = plt.figure()
        for service, qs in queues.items():
            for name, vals in qs.items():
                # queue_ax = None
                # * If dimensions is (1,1) then that means we only have numbers for queues
                # * That can happen if a module's execution is conditional (e.g. requires detections)
                # * Handle that case as then axs is NOT a list
                # if dimensions == (1, 1):
                #     queue_ax = axs
                # else:
                #     queue_ax = axs[plt_cnt_x, plt_cnt_y]
                plt.plot(vals, label="{}_{}".format(service, name))
                plt.legend()
                # plt.set_ylim([0, 200])
    except IndexError as e:
        print(e)
    return fig_plt, plt_metrics, queue_ax, plt_queues


def change_btn_color(ready):
    if ready:
        colour = "#539200e6"
    else:
        colour = "#f63366"

    st.markdown("""<style>
    #root > div:nth-child(1) > div > div > div > div > section.css-1lcbmhc.e1fqkh3o0 > div.css-hby737.e1fqkh3o1 > div.block-container.css-1dnb06j.eknhn3m2 > div:nth-child(1) > div:nth-last-child(3) > div > button {
    background-color: """ + colour + """;color: white;
    }
    </style>
    """, unsafe_allow_html=True)

#root > div:nth-child(1) > div > div > div > div > section.css-1lcbmhc.e1fqkh3o0 > div.css-hby737.e1fqkh3o1 > div.block-container.css-1dnb06j.eknhn3m2 > div:nth-child(1) > div:nth-child(11) > div > button