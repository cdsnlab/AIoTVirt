import os

import numpy as np
import pandas as pd
from pymongo import MongoClient
import json

#! NOT COMPLETE YET!
#* plots are not GPS

client = MongoClient('localhost', 27017)
db = client['koln']
mdb = db['plots'] 

koln_avg_lat = 50.977943
koln_avg_lon = 6.918564

#* koln dataset
# More precisely, each line of the trace contains the time (with 1-second granularity), the vehicle identifier, its position on the two-dimensional plane (x and y coordinates in meters) and its speed (im meters per second)
# time(s), id, position, 
# 23924 1395234 12297.386758682234 15881.641281434131 1.94

df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/koln.tr/koln.tr',error_bad_lines=False)
print(df.head())
# trajectories=df['POLYLINE'].tolist()

# MAXPLOTS = 10000

# count=0
# for i in range (len(trajectories)):
    
#     latplots = []
#     lonplots = []

#     allplots = []
#     res=json.loads(trajectories[i]) 
#     print(len(res))
#     for j in range(len(res)):
#         allplots.append([res[j][1], res[j][0]])
#         latplots.append(res[j][1])
#         lonplots.append(res[j][0])
#     row = {"index": str(count), "lon": lonplots, "lat": latplots, "both": allplots}
#     mdb.insert_one(row)
#     print(row)

#     count+=1
#     if count > MAXPLOTS: #* so it doesn't break :( 
#         break
