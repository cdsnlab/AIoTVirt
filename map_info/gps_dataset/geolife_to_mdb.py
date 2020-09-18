import os

import numpy as np
import pandas as pd
from pymongo import MongoClient
import json

#! NOT COMPLETE YET!
#* use this link to parse!
#* https://heremaps.github.io/pptk/tutorials/viewer/geolife.html


allfiles=[]

client = MongoClient('localhost', 27017)
db = client['geolife']
mdb = db['plots'] 

geolife_avg_lat = 39.90745772086431
geolife_avg_lon = 116.35544946451571
#* geolife dataset
# /home/spencer1/samplevideo/gps_datasets/geolife_dataset/Data/XXX/Trajectory/YYYY.plt
#df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/geolife_dataset/Data/')
df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/geolife_dataset/Data/')


for (path, dir, files) in os.walk('/home/spencer1/samplevideo/gps_datasets/geolife_dataset/Data/'):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.plt':
            print("%s/%s" % (path, filename))
            allfiles.append(path+"/"+filename)

for file in allfiles:
    if os.stat(file).st_size!=0:
        df=pd.read_csv(file, head)

MAXPLOTS = 10000

count=0
for i in range (len(trajectories)):
    
    latplots = []
    lonplots = []

    allplots = []
    res=json.loads(trajectories[i]) 
    print(len(res))
    for j in range(len(res)):
        allplots.append([res[j][1], res[j][0]])
        latplots.append(res[j][1])
        lonplots.append(res[j][0])
    row = {"index": str(count), "lon": lonplots, "lat": latplots, "both": allplots}
    mdb.insert_one(row)
    print(row)

    count+=1
    if count > MAXPLOTS: #* so it doesn't break :( 
        break
