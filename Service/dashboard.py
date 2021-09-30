import time
import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import math

# from line_profiler import LineProfiler
# profile = LineProfiler()
# import atexit
# atexit.register(profile.print_stats)

plt.style.use("ggplot")


device_urls = ["localhost:5000", "localhost:5000", "localhost:5000",
               "localhost:5000", "localhost:5000", "localhost:5000"]

device_urls = ["localhost:33331", "localhost:33332"]

quantile = st.slider(label="Slider for quantile filtering",
                     min_value=0.0, max_value=1.0, value=0.95)

cols = st.columns(3)

start = st.checkbox("Start monitoring/pipeline")

device_counter = 0
devices = {}
for col in cols:
    try:
        for i in range(2):
            col.subheader("Device " + str(device_counter))
            address = col.text_input(
                "Server url:port", value=device_urls[device_counter], key="device_{}_address".format(device_counter))
            plt_chart = col.empty()
            devices[device_counter] = {
                "address": address,
                "chart": plt_chart
            }

            device_counter += 1
    except IndexError:
        pass


def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/3), 3

def rec(values):
    vals = np.array(values)
    filtered = vals
    try:
        filtered = vals[vals < np.quantile(vals, quantile)]
    except:
        pass
    # lines["{}_{}".format(service, metric)] = filtered
    return filtered

# @profile
def plot(address, chart):
    data = requests.get("http://{}/metrics".format(address)).json()

    # queues = requests.get("http://{}/queue_capacity".format(address)).json()

    lines = {}
    
    try:
        for service, metrics in data.items():
            for metric, values in metrics.items():
                # if metric == "remote":
                if "list" in values:
                    # pass
                    filtered = rec(values["list"])
                    if filtered.any():
                        lines["{}_{}".format(service, metric)] = filtered
                else:
                    for m, v in values.items():
                        if "list" in v:
                            filtered = rec(v["list"])
                            if filtered.any():
                                lines["{}_{}_{}".format(service, metric, m)] = filtered
                        else:
                            for mm, vv in v.items():
                                # print(vv["list"])
                                # lines["{}_{}_{}_{}".format(service, metric, m, mm)] = rec(vv["list"])
                                filtered = rec(vv["list"])
                                if filtered.any():
                                    lines["{}_{}_{}_{}".format(service, metric, m, mm)] = filtered

                        # print(m)

                # vals = np.array(values["list"])
                # filtered = vals
                # try:
                #     filtered = vals[vals < np.quantile(vals, quantile)]
                # except:
                #     pass
                # lines["{}_{}".format(service, metric)] = filtered

        # * +1 is for the queue capacities
        axes = choose_subplot_dimensions(len(lines) + 1)
        fig_plt, axs = plt.subplots(nrows=axes[0], ncols=axes[1])

        plt_cnt_x, plt_cnt_y = 0, 0
        for label, line in lines.items():
            axs[plt_cnt_x, plt_cnt_y].plot(line, label=label)
            axs[plt_cnt_x, plt_cnt_y].legend()
            plt_cnt_y += 1
            if plt_cnt_y == axes[1]:
                plt_cnt_y = 0
                plt_cnt_x += 1

        # TODO NEed to get queue sizes and plot them in ONE chart
        # for service, qs in queues.items():
        #     for name, vals in qs.items():
        #         axs[plt_cnt_x, plt_cnt_y].

        chart.write(fig_plt)
        plt.close(fig_plt)
    except IndexError as e:
        print(e)


while start:
    time.sleep(1)
    if not start:
        break
    for device in devices.keys():
        plot(**devices[device])
