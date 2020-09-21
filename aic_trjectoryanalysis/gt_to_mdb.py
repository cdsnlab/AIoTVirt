'''
[DONE]
this program writes all [ID, camera, x, y]
'''
import os, sys, time
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from pymongo import MongoClient

#* connect to mongodb
client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
mdb = db['draw_traces']

allfiles=[]

def get_camera_id(filename):
    #print(filename)
    items = filename.split('/')
    for item in items:
        if item.startswith("c0"):
            return item

#* iterate through folder to find all gt.txt files.
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
        items = defaultdict(lambda: defaultdict(list))
        camid = get_camera_id(file)
        for index, row in df.iterrows(): #[frame, ID, left, top, width, height, 1, -1, -1, -1]
            #* however many ids there are.            
            centx = row[2] + (row[4] / 2)
            centy = row[3] + (row[5] / 2)

            items[row[1]]["x"].append(centx) #id, x: list, y: list
            items[row[1]]["y"].append(centy)
            #print(row[1])

            
        #print(items[row[1]]["x"], items[row[1]]["y"])   
        #print(len(items))
        for i in (items):
            #print(i)
            inputrow = {"uid": str(i),"camid": str(camid), "x": items[i]["x"], "y": items[i]["y"]}
            #print(inputrow)
            mdb.insert_one(inputrow)
            count+=1

