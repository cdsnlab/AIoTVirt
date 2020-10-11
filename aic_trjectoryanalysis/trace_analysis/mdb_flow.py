"""
[DONE]
we first save the traces by uid, section, camid, trace for all files, where trace contains [start, end]
#? we don't need to seperate train folder & validation folder since we can pick what to use in our simulation 
#! if you have time, change this to reading from gt db instead of reading the files :S
"""

import os, sys, time
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
mdb = db['uid_traces']

allfiles=[]

def get_camid_section(filename):
    camid, section = None, None
    items = filename.split('/')
    for item in items:
        if item.startswith("c0"):
            camid = item
        if item.startswith("S"):
            section = item
    return camid, section

#* iterate through folder to find all gt.txt files "in train folder" 
for (path, dir, files) in os.walk ("/home/spencer1/samplevideo/AIC20_track3_MTMC/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == ".txt" and "gt" in os.path.splitext(filename)[0]:
            #print("%s/%s" % (path, filename))
            allfiles.append(path+"/"+filename)

count=0

for file in tqdm(allfiles):
    xplots = []
    yplots = []
    #print(count)
    if os.stat(file).st_size != 0:
        df = pd.read_csv(file, header=None, delimiter=',')
        items = defaultdict(list)
        camid, section = get_camid_section(file)
        for index, row in df.iterrows(): #[frame, ID, left, top, width, height, 1, -1, -1, -1]
            #* however many ids there are.            
            items[row[1]].append(str(row[0]))

            
        #print(items[row[1]]["x"], items[row[1]]["y"])   
        #print(len(items))
        for i in (items):
            #print(items[i][0], items[i][-1])
            inputrow = {"id": str(i), "sector": str(section), "camid": str(camid), "trace": list(items[i]), "start": int(items[i][0]), "end": int(items[i][-1])}
            print(inputrow)
            mdb.insert_one(inputrow)
            count+=1
        #print(items)

