'''
[DONE]
this program finds the matchings between cameras by reading the db.
- which cameras has relationship btw each other. (spatio)
- how much time it takes for a car to travel from one to another. (temporal)
INPUT: All traces with the same UID
OUTPUT: save spatio-temporal relationship btw two cameras
1) camera relationships
2) Time btw each pair of cameras

!!!Difference between spatio_temporal_all_relationship.py is that this one prunes overlapping cases.
'''
CAMERA_SECTION ={ #* S05 is a mix of cameras from Section 03 to 04
    "S01": {"c001", "c002", "c003", "c004", "c005"},
    "S02": {"c006", "c007", "c008", "c009"},
    "S03": {"c010", "c011", "c012", "c013", "c014", "c015"},
    "S04": {"c016", "c017", "c018", "c019", "c020", "c021", "c022", "c023", "c024", "c025", "c026", "c027", "c028", "c029", "c030", "c031", "c032", "c033", "c034", "c035", "c036", "c037", "c038", "c039", "c040"},
    "S06": {"c041", "c042", "c043", "c044", "c045", "c046"} #! does not contain gt
}

import os, sys, time
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
iddb = db['uid_traces']
stdb = db['spatio_temporal']
MAXUID = 0


allfind = iddb.find() #* get all UIDs
for trace in allfind: 
    #print(trace["uid"])
    if int(trace["uid"]) >MAXUID:
        MAXUID = int(trace["uid"])

def remove_samestarts(doc): # incase they are seen in multiple "start" views
    while len(doc) > 1:
        for i in range(len(doc)-1): # dropping traces which has to the same starting time
            if int(doc[i]['start']) == int(doc[i+1]['start']):
                if int(doc[i]['end']) > int(doc[i+1]['end']):
                    #print(">dropping {}, {}, {}".format(i+1, doc[i+1], doc))
                    doc.remove(doc[i+1])
                    break
                else:
                    #print("<dropping {}, {}, {}".format(i, doc[i], doc))
                    doc.remove(doc[i])
                    break
            else:
                continue
        add=0
        for i in range(len(doc)-1): # if all don't start at the same time, leave loop.
            #print(add, len(doc)-1)
            if int(doc[i]['start']) != int(doc[i+1]['start']):
                add+=1
            if add == len(doc)-1:              
                return doc
    return doc
            
def remove_containing(doc): #* we don't know that the next starting point is early or not...
    #print(doc)
    while len(doc) > 1:
        for i in range(len(doc)-1): 
        # 간단하게 0번 시작 포인트 - 종료 포인트, + 1번 시작 포인트 가 2번 종료 포인트 보다 이른지 확인하면 됨. TODO: 1번 시작 포인트가 0번 종료 시점보다 늦으면 패스.
            if int(doc[i]['start']) < int(doc[i+1]['start']): 
                if int(doc[i+1]['start']) <= int(doc[i]['end']): #* this pair is overlapping
                    if int(doc[i+1]['end']) <= int(doc[i]['end']): #* 
                    #if ( (int(doc[i+1]['end'])- int(doc[i+1]['start'])) + int(doc[i]['start']) )<= int(doc[i]['end']):
                        #print(">removed {}, {}, {}".format(i, doc[i+1], doc))
                        doc.remove(doc[i+1])
                        break
        add=0
        for i in range(len(doc)-1):
            if int(doc[i]['end']) < int(doc[i+1]['end']):
                add+=1
            if add == len(doc)-1:
                return doc
    return doc
        
def pairwise(doc1, doc2, pairs):
    pairs[(doc1['camid'], doc2['camid'])].append(int(doc2['start'])-int(doc1['end']))
    return pairs

pairs = defaultdict(list)

for i in range(1, MAXUID):
    myquery = {"uid": str(i)}
    doc = list(iddb.find(myquery,  {"uid":1, "camid": 1, "start":1, "end": 1} ).sort([("start", 1)]))
    #print("i: {}".format(i))
    
    doc = remove_samestarts(doc) #* removes all "same start frame candidates". That is, same starting frame numbers!
    doc = remove_containing(doc) #* removes all "containing traces". That is, a camera's covered by another camera's view which has long duration

    for j in range(len(doc)-1):
        pairs = pairwise(doc[j], doc[j+1], pairs)
        
        #print(pairs) 

for k in pairs:
    inputrow = {"pairs": k, "src": k[0], "dst": k[1], "temporal": pairs[k]}
    print(inputrow)
    #stdb.insert_one(inputrow)