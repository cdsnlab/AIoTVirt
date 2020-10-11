"""
[DONE]
collects IDs found in training / validation and saves a list of cameras into mongo db.
"""

import os, time
from tqdm import tqdm
import pandas as pd
from pymongo import MongoClient
import random

client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
gtdb = db['gt']
idlistsdb=db['idlists'] #* contains all ids in either  train or validate dataset. 


sectors = ["S01", "S03", "S04"]
for sec in sectors: 
    training=[]
    camlist=[]
    lookup = {"sector": sec}
    doc = list(gtdb.find(lookup, {"_id": 0, "sector": 1, "id":1, "camera":1}))
    for d in doc:
        if d['id'] not in training:
            training.append(d['id'])
            #print(d)    
        if d['camera'] not in camlist:
            camlist.append(d['camera'] )
    #print("in train: {}".format(training))
    line = {"label": "training", "sectors": sec, "camera": camlist, "ids": training}
    print(line)
    idlistsdb.insert_one(line)

validatesector = ["S02", "S05"]
for vsec in validatesector:
    validating=[]
    camlist=[]
    vlookup = {"sector": vsec}
    vdoc = list(gtdb.find(vlookup, {"_id": 0, "sector": 1, "id":1, "camera":1}))
    #print(vdoc)
    for vd in vdoc:
        #print(vd['camera'])
        if vd['id'] not in validating:
            validating.append(vd['id'])
        if vd['camera'] not in camlist:
            camlist.append(vd['camera'] )

    line = {"label": "validation", "sectors": vsec, "camera":camlist, "ids": validating}
    print(line)
    idlistsdb.insert_one(line)


