import os
import streamlit as st
import numpy as np

import plotly.graph_objects as go
from pymongo import MongoClient

MAXPLOTS = 500

client = MongoClient('localhost', 27017)
db = client['roma']
mdb = db['plots']  #* collection name
allpoints = mdb.find({}) #* find all instances from the collection.

roma_avg_lat = 41.9004122222972
roma_avg_lon = 12.4728368659278

mapbox_access_token = open(".mapbox_token").read()

gpscoordinates = {}

for i, doc in enumerate(allpoints):
    #print(doc)
    gpscoordinates[i] = doc


fig = go.Figure()
for i in range(MAXPLOTS):
    fig.add_trace(go.Scattermapbox(
        lat=gpscoordinates[i]['lat'], 
        lon=gpscoordinates[i]['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=3
        ),
    ))

fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=roma_avg_lat,
            lon=roma_avg_lon
        ),
        pitch=0,
        zoom=10.9
    ),
)

st.plotly_chart(fig)

