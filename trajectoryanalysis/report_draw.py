import numpy as np
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def click_callback(trace, points, state):
    print(trace)
    print(points)
    print(state)

def c():
    print(111)
    
data = pd.read_excel("/home/spencer1/AIoTVirt/trajectoryanalysis/results/scenario_newsimdata_prop_variation.xlsx", sheet_name=None)

pp = st.selectbox("Preprocessing", ["all", "last", "ed", "irw", "sw-o"])
vl = st.selectbox("VL - seq length", ["all", "15", "30"])
time_est = st.selectbox("Time Estimation Model", ["all", "ResNet", "rf", "dt", "svm", "hcf"])
precisions, accuracies = [], []
sheet_names = []
for sheet_name, df in data.items():
    # print(df)
    if pp not in sheet_name and pp != "all":
        continue
    if vl not in sheet_name and vl != "all":
        continue
    if time_est not in sheet_name and time_est != "all":
        continue
    try:
        avg_prec = df["precision"].mean()
        avg_acc = df["accuracy"].mean()
        precisions.append(avg_prec)
        accuracies.append(avg_acc)
        sheet_names.append(sheet_name)
    except KeyError:
        pass
fig = go.FigureWidget()
a = fig.add_bar(name="precision", x=sheet_names, y=precisions)
# bar = go.Bar(name="precision", x=sheet_names, y=precisions)
b = fig.add_bar(name="accuracy", x=sheet_names, y=accuracies)    
# bar = go.Bar(name="accuracy", x=sheet_names, y=accuracies)
st.plotly_chart(fig, use_container_width=True)

fig2 = go.FigureWidget()
single_run = st.selectbox("Select sheet name", sheet_names)
acc_run = data[single_run]["accuracy"]
prec_run = data[single_run]["precision"]
fig2.add_bar(name="precision", x=list(range(len(prec_run))), y=prec_run)
fig2.add_bar(name="accuracy", x=list(range(len(acc_run))), y=acc_run)

st.plotly_chart(fig2, use_container_width=True)
