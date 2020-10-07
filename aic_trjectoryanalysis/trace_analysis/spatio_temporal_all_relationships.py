'''
[DONE]
this program finds the matchings between cameras by reading the db.
- which cameras has relationship btw each other. (spatio)
- how much time it takes for a car to travel from one to another. (temporal)
INPUT: All traces with the same UID
OUTPUT: save spatio-temporal relationship btw two cameras
1) camera relationships
2) Time btw each pair of cameras

!!!Difference between spatio_temporal.py is that this one insert all possible .
'''
import networkx as networkx
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
iddb = db['uid_traces']
stalldb = db['st_all']
MAXUID = 0

CAMERA_SECTION ={ #* S05 is a mix of cameras from Section 03 to 04
    "S01": {"c001", "c002", "c003", "c004", "c005"},
    "S02": {"c006", "c007", "c008", "c009"},
    "S03": {"c010", "c011", "c012", "c013", "c014", "c015"},
    "S04": {"c016", "c017", "c018", "c019", "c020", "c021", "c022", "c023", "c024", "c025", "c026", "c027", "c028", "c029", "c030", "c031", "c032", "c033", "c034", "c035", "c036", "c037", "c038", "c039", "c040"},
    "S06": {"c041", "c042", "c043", "c044", "c045", "c046"} #! does not contain gt
}

allfind = iddb.find() #* get all UIDs
for trace in allfind: 
    #print(trace["uid"])
    if int(trace["uid"]) >MAXUID:
        MAXUID = int(trace["uid"])

def get_sector(camid):
    for cs in CAMERA_SECTION:
        if camid in CAMERA_SECTION[cs]:
            return cs

def pairwise(doc1, doc2, time, pairs):
    pairs[(doc1['camid'], doc2['camid'])].append(time)
    return pairs

pairs = defaultdict(list)

for i in range(1, MAXUID):
    myquery = {"uid": str(i)}
    doc = list(iddb.find(myquery,  {"uid":1, "camid": 1, "start":1, "end": 1} ).sort([("start", 1)]))
    print("i: {}, doc: {}".format(i, doc))
    
    count = 0
    length = len(doc)
    while count <= length-1:
        for l in range(length):
            print(count, l, doc[count], doc[l])
            time = 0
            if count == l:
                continue
            if int(doc[l]['start']) < int(doc[count]['end']):
                if int(doc[l]['end']) <= int(doc[count]['end']):
                    continue
                elif int(doc[l]['end']) > int(doc[count]['end']):
                    time = (int(doc[count]['end']) - int(doc[l]['start']))*-1
                    
                    pass
            
            if int(doc[l]['start']) > int(doc[count]['end']):
                time = int(doc[l]['start']) - int(doc[count]['end'])
                pass
            if time != 0:
                pairs = pairwise (doc[count], doc[l], time,  pairs)
            print (time)

        count +=1


for k in pairs:
    src_sector = get_sector(k[0])
    dst_sector = get_sector(k[1])
    inputrow = {"pairs": k, "src": k[0], "src_s":src_sector, "dst": k[1], "dst_s": dst_sector, "temporal": pairs[k]}
    print(inputrow)
    #stalldb.insert_one(inputrow)
