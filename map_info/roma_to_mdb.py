import os
import numpy as np
import pandas as pd
import datetime
from pymongo import MongoClient
from tqdm import tqdm
import streamlit as st
import plotly.graph_objects as go


#! 3 already give upto 2500 unique traces
MAXIDS=3
#! PROBLEM 
#! we don't know if this is one trip or not by seperating the trip by time difference between two plots...
MAXTIMEDIFF = 15 # seperation of time difference between two trips.

client = MongoClient('localhost', 27017)
db = client['roma']
mdb = db['plots'] 

#* each row contains an id + (lat, lon) plots -> we need to create a list which keeps on adding plots until it reaches the end.
df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/roma_taxi/taxi_february.txt', error_bad_lines=False, header=None,delimiter=";")

roma_avg_lat = 41.9004122222972
roma_avg_lon = 12.4728368659278

# 1) iterate the whole file once, append gps coordinate on every ID occurances -> if this fails, we lose everything 
# 2) iterate len(ID) times but only picking its own ID and coordinates every iteration -> we can stop anytime we want.

# print(df[df[0].isin([317])])
latplots=[]
lonplots=[]
allplots=[]
prevtime = None
tripid = 0
for j in tqdm(range(MAXIDS)):
    for index, row in (df[df[0].isin([j])].iterrows()):
        #* need to check if the last time slice was longer than 20seconds or more
        # if yes, seperate n increase ID count and collect latplots, lonplots to blank list
        curtime = datetime.datetime.strptime(row[1][:-3], '%Y-%m-%d %H:%M:%S.%f')
        if prevtime !=None:
            #print((curtime-prevtime).seconds)
            if (curtime -prevtime).seconds > MAXTIMEDIFF:  # diff trip!
                inputrow = {"index": str(j)+"_"+str(tripid), "lon": lonplots, "lat": latplots, "both": allplots}
                print(str(j)+"_"+str(tripid))
                mdb.insert_one(inputrow)
                #print(inputrow)
                tripid+=1
                latplots=[]
                lonplots=[]
                allplots=[]
                # print(curtime)
            item = row[2].replace('POINT','')
            item = item.replace('(', '')
            item = item.replace(')', '')
            latlon = item.split(' ')

            latplots.append(latlon[0])
            lonplots.append(latlon[1])
            allplots.append([latlon[0], latlon[1]])

        prevtime = curtime
    tripid=0
    latplots=[]
    lonplots=[]
    allplots=[]
    prevtime = None
