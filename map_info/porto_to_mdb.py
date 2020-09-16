import os

import numpy as np
import pandas as pd
from pymongo import MongoClient
import json


client = MongoClient('localhost', 27017)
db = client['porto']
mdb = db['plots'] 

porto_avg_lat = 41.156226
porto_avg_lon = -8.571681

#* porto full dataset
df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/porto_dataset/train.csv/train.csv', error_bad_lines=False)

#* porto 300 dataset
#df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/porto_dataset/Porto_taxi_data_test_partial_trajectories.csv')

trajectories=df['POLYLINE'].tolist()

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
