import os
import streamlit as st
import numpy as np

import plotly.graph_objects as go
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['tdrive']
mdb = db['plots']  #* collection name

tdrive_avg_lat = 39.90745772086431
tdrive_avg_lon = 116.35544946451571


#* for streamlit built in func ---
# map_data = pd.DataFrame(
#     allplots,
#     columns=['lat', 'lon'],
#     )
# st.map(map_data, zoom=12)

#* for plotly + streamlit ---
mapbox_access_token = open(".mapbox_token").read()

#TODO loop this to create multiple ones.
fig = go.Figure(go.Scattermapbox(
    lat=latplots,
    lon=lonplots,    
    mode='markers',
    marker=go.scattermapbox.Marker(
        size = 3
    ),
))

fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=tdrive_avg_lat,
            lon=tdrive_avg_lon
        ),
        pitch=0,
        zoom=9
    ),
)

st.plotly_chart(fig)