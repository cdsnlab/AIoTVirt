import os
import streamlit as st
import numpy as np

import plotly.graph_objects as go
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['sf']
mdb = db['plots']  #* collection name

sf_avg_lat = 37.765963
sf_avg_lon =  -122.433639

MAXPLOTS = 500

#* for plotly + streamlit ---
mapbox_access_token = open(".mapbox_token").read()

gpscoordinates = {}
for i in range(MAXPLOTS):
    myquery = {"index": str(i)}
    # doc contains lats and lons 
    doc=list(mdb.find(myquery, {"_id":0, "lat": 1, "lon": 1}))
    
    gpscoordinates[i]=doc

fig = go.Figure()
for i in range(MAXPLOTS):
    fig.add_trace(go.Scattermapbox(
        lat=gpscoordinates[i][0]['lat'], 
        lon=gpscoordinates[i][0]['lon'],
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
            lat=sf_avg_lat,
            lon=sf_avg_lon
        ),
        pitch=0,
        zoom=10
    ),
)

st.plotly_chart(fig)