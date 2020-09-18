import os
import streamlit as st
import numpy as np

import plotly.graph_objects as go
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['porto']
mdb = db['plots']  #* collection name

porto_avg_lat = 41.156226
porto_avg_lon = -8.611681

MAXPLOTS = 500


#* for streamlit built in func ---
# map_data = pd.DataFrame(
#     allplots,
#     columns=['lat', 'lon'],
#     )
# st.map(map_data, zoom=12)

#* for plotly + streamlit ---
mapbox_access_token = open(".mapbox_token").read()

gpscoordinates = {}
for i in range(MAXPLOTS):
    myquery = {"index": str(i)}
    # doc contains lats and lons 
    doc=list(mdb.find(myquery, {"_id":0, "lat": 1, "lon": 1}))
    print(doc)
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
            lat=porto_avg_lat,
            lon=porto_avg_lon
        ),
        pitch=0,
        zoom=10.9
    ),
)

st.plotly_chart(fig)

