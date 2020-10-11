'''
[DONE]
this program visualizes gt traces to the scene
'''
import os, sys, time
from tqdm import tqdm
import streamlit as st
from PIL import Image

import plotly.graph_objects as go
from pymongo import MongoClient

#* connect to mongodb
client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
mdb = db['draw_traces']

data_load_state = st.text('Loading data...')

def get_camera_id(filename):
    #print(filename)
    items = filename.split('/')
    for item in items:
        if item.startswith("c0"):
            return item[:4] #* only returns c000 without .jpg

allfiles=[]
allfiles_names =[]
for (path, dir, files) in os.walk("/home/spencer1/samplevideo/AIC20_track3_MTMC/scene"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == ".jpg":
            allfiles.append(path + "/" + filename)
            
            allfiles_names.append(filename[:4])
#print(allfiles_names)
cam_number = st.selectbox("Camera number", allfiles_names)

data_load_state.text('Loading data...done!')
st.title("AIC traces on the map")

widget = go.Figure()

#print ("/home/spencer1/samplevideo/AIC20_track3_MTMC/scene/"+str(cam_number)+".jpg")
myquery = {"camid": str(cam_number)}
doc = list(mdb.find(myquery, {"_id": 0, "camid":1, "x":1, "y":1}))
pilimage = Image.open("/home/spencer1/samplevideo/AIC20_track3_MTMC/scene/"+str(cam_number)+".jpg")

# Add images
widget.add_layout_image(
        dict(
            source=pilimage,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=pilimage.width,
            sizey=pilimage.height,
            sizing="stretch",
            opacity=1,
            layer="below")
)

# widget.add_layout_image(
#     dict(
#         source="/home/spencer1/samplevideo/AIC20_track3_MTMC/scene/"+str(cam_number)+".jpg",
#         opacity=1.0,
#         sizex = 1000,
#         sizey = 1000
#     )
# )
widget.update_layout(template="plotly_white")
#st.image("/home/spencer1/samplevideo/AIC20_track3_MTMC/scene/"+str(cam_number)+".jpg", width=None)
for i in range(len(doc)):
    widget.add_trace(go.Scatter(x=doc[i]["x"],y=doc[i]["y"]))
widget.update_yaxes(autorange="reversed")
st.plotly_chart(widget, use_container_width=True)


#* 2) load traces
# for file in tqdm(allfiles): #* iterate all bg scenes 
#     trace_per_cam={}
#     camid = get_camera_id(file)
    
#     myquery = {"camid": str(camid)}
#     doc = list(mdb.find(myquery, {"_id": 0, "camid":1, "x":1, "y":1}))
    
#     #print(camid, len(doc))
#     for i in range(len(doc)): #* iterate all bg plots
#         trace_per_cam[i] = doc[i]


#* 3) print traces