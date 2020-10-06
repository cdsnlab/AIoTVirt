"""
[Prog]
INPUT: all gt files
OUTPUT: average number of frames processed, average number of reid-ed objects (vehicles), average number of objects 
This program finds the number of cameras which needs to be turned on to track a vehicle. GIVEN that all re-id is PERFECT.
#?Each simulation ends when the vehicle is no long in the network. Should we give it additonal time (e.g. 100 frames = 10seconds) ???
"""
#TODO 
# 1) pick a target id to track.
# 2) if found, find next cameras, next trackable candidates 

import os, sys, time
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from pymongo import MongoClient

def get_camid_section(filename):
    camid, section = None, None
    items = filename.split('/')
    for item in items:
        if item.startswith("c0"):
            camid = item
        if item.startswith("S"):
            section = item
    return camid, section

def get_candidates(framenumber, af, ids): # 같은 framenumber에 있는 모든 물체 반환. 없으면 상관 없음. 
    items = []
    for file in af:
        if os.stat(file).st_size != 0:
            df = pd.read_csv(file, header=None, delimiter=',')
            allobjects = df[(df[0])==framenumber] # all vehicles in that framenumber
            sfn = df[(df[0]==framenumber)&(df[1]==ids)] # target vehicle in that framenumber
            camid, section = get_camid_section(file) 
            if len(allobjects) !=0:
                items.append((camid, section, len(allobjects), sfn.values.tolist()))
    #print(items)
    return items
def checkifsame(status, items):
    for item in items:
        #print(item, status)
        if str(status) == str(item[0]):
            return True
    return False
    
def get_maxiter(ids):
    dblookup = {"uid": str(ids)}
    doc=list(iddb.find(dblookup, {"_id":0, "camid":1, "trace":1}))
    i=0
    maxlen = 0
    for d in doc:
        #print(d['trace'][-1])
        if int(d['trace'][-1]) > maxlen:
            #print(maxlen)
            maxlen = int(d['trace'][-1])
        i+=1
    print ("Maximum iteration:{}".format(maxlen))
    return maxlen

client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
iddb = db['uid_traces']
stdb = db['spatio_temporal']

#* iterate through folder to find all gt.txt files. find maximum number of frames to look through
#? keep it here for the time being: we don't want our system to iterate more than needed
allfiles=[]
MAXLINE=0
for (path, dir, files) in os.walk ("/home/spencer1/samplevideo/AIC20_track3_MTMC/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == ".txt" and "gt" in os.path.splitext(filename)[0]:
            #print("%s/%s" % (path, filename))
            allfiles.append(path+"/"+filename)
            num_lines = sum(1 for line in open(path+"/"+filename))
            
            if MAXLINE < num_lines:
                MAXLINE = num_lines

#* iterate through DB to find maximum number of ids.
MAXUID = 0
allfind = iddb.find() #* get all UIDs
for trace in allfind: 
    #print(trace["uid"])
    if int(trace["uid"]) >MAXUID:
        MAXUID = int(trace["uid"])

count=0
print("MAXUID: {}".format(MAXUID))

for ids in tqdm(range(1,MAXUID)): # for all ids
    MAXITER = get_maxiter(ids)
    #print(doc['trace'])
    items = None
    cumulative_processed =0 # cumulative number of frames looked at. this include frames that may not contain target vehicle in the view.
    cumulative_reid = 0 # cumulative number of objects being reid'd
    STATUS = None # (prev status, cur status, camera number) # to get if it has 
    for framenumber in range(1, MAXITER): # run through all gt files to find it at that framenumber. 
        print("[INFO] framenumber: {}, STATUS {}".format(framenumber, STATUS))
        items = get_candidates(framenumber, allfiles, ids) # (camid, sector, number of vehicles in the view, location of the target vehicle in the view)
        #TODO1: if found (beginning): count the number of objects being re-id in that camera. Keep status updated (tracking T/F, camera number)
        #TODO2: if in transition: count the number of frames being processed 
        tvcandidates = []
        for item in items:
            if item[3]:
                tvcandidates.append(item)
        if tvcandidates: # its TRK
            print(tvcandidates) 
            if STATUS == None: 
                STATUS = ("NF", "TRK", tvcandidates[0][0])
            elif STATUS[1] == "TRK":
                if checkifsame(STATUS[2], tvcandidates): #* its still tracking from the same camera
                    continue
                else: #* its being seen from a different camera. Lets switch to it
                    STATUS = ("TRK", "TRK", tvcandidates[0][0])
                    print("[INFO] changing camera to {}".format(tvcandidates[0][0]))
            cumulative_processed+=1
            cumulative_reid+=1
        else: # its either NF/L/TRS
            if STATUS==None: #* Never seen before
                STATUS=("NF","NF", -1)
                print("[INFO] Never seen before...")
            elif STATUS[1]=="TRK": #Its been trking 
                STATUS = ("TRK", "TRS", -1)
                print("[INFO] IN Transition...")

            #TODO need to add previous status in STATUS so that it knows that its been TRK not NF. 
        #! check if there its the first occurance. IF there are more than one at the same period, pick one at random.
        #! check if its being notfound(nf)/tracked(trk)/lost(l)/transition(trs)
        #print (tvcandidates)
        # if tvcandidates:
        #     if STATUS == None:
        #         STATUS = (True, )
        '''
        for item in items:
            if item[3] : #* if there is TV
                if STATUS == None: #* its the beginning
                    STATUS = (True, item[0])
                    cumulative_processed+=1
                    cumulative_reid+=item[2]
                    break
                else: #* its seen TV before.
                    # check if it is the same ID. #TODO add it to a temp list, and then see if its there.
                    if item[1] == STATUS[1]: # if ID, continue tracking
                        print("continue tracking")
                        cumulative_processed+=1
                        cumulative_reid+=item[2]
                        break
                    else: # if it isn't, its transfered to another. #! what if its not ordered???
                        STATUS[1] = item[1]


            else: # does not have TV
                if STATUS == None: # keep searching
                    continue
                else: # its lost.
                    STATUS[1] = -1
        '''
                    

# for file in tqdm(allfiles):
#     xplots = []
#     yplots = []
#     #print(count)
#     if os.stat(file).st_size != 0:
#         df = pd.read_csv(file, header=None, delimiter=',')
#         items = defaultdict(lambda: defaultdict(list))
#         camid, section = get_camid_section(file)
#         for index, row in df.iterrows(): #[frame, ID, left, top, width, height, 1, -1, -1, -1]
#             #* however many ids there are.            
#             centx = row[2] + (row[4] / 2)
#             centy = row[3] + (row[5] / 2)

#             items[row[1]]["x"].append(centx) #id, x: list, y: list
#             items[row[1]]["y"].append(centy)
#             #print(row[1])
