"""
[Prog]
INPUT: all gt files
OUTPUT: average number of frames processed, average number of reid-ed objects (vehicles), average number of objects 
This program finds the number of cameras which needs to be turned on to track a vehicle. GIVEN that all re-id is PERFECT.
#?Each simulation ends when the vehicle is no long in the network. Should we give it additonal time (e.g. 100 frames = 10seconds) ???
"""

import os, time
from tqdm import tqdm
import pandas as pd
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

def checkifsame(camid, items):
    for item in items:
        if str(camid) == str(item[0]):
            if item[3]:
                return True
    return False
    
def searchnextcamcandidates(src):
    stitems={}
    srclookup={"src": str(src)}
    doc=list(stdb.find(srclookup, {"_id":0, "src":1, "dst": 1, "temporal":1}))
    for d in doc:
        stitems[d['dst']] = (min(d['temporal']), max(d['temporal']))
        #sum(d['temporal']) / len(d['temporal'])
    #print(stitems)
    return stitems
    #what should this return?   
    # next cameras, earliest possible start time.

def compare_both(predicted_cam, all_things):
    for i in predicted_cam:
        for j in all_things:
            #print(i,j)
            if j[0] == i:
                if j[3]:
                    return i
    return -1

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

allfiles=[]
all_cameras = []
for (path, dir, files) in os.walk ("/home/spencer1/samplevideo/AIC20_track3_MTMC/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == ".txt" and "gt" in os.path.splitext(filename)[0]:
            allfiles.append(path+"/"+filename)
            #print("%s/%s" % (path, filename))
            camid, section = get_camid_section(path+"/"+filename)
            all_cameras.append(camid)
print(all_cameras)


#* iterate through DB to find maximum number of ids.
MAXUID = 0
allfind = iddb.find() #* get all UIDs
for trace in allfind: 
    #print(trace["uid"])
    if int(trace["uid"]) >MAXUID:
        MAXUID = int(trace["uid"])

count=0
print("MAXUID: {}".format(MAXUID))

for ids in tqdm(range(7,MAXUID)): # for all ids
    MAXITER = get_maxiter(ids)
    #print(doc['trace'])
    items = None
    cumulative_processed =0 # cumulative number of frames looked at. this include frames that may not contain target vehicle in the view.
    cumulative_reid = 0 # cumulative number of objects being reid'd
    STATUS = ("NF", "NF", -1) # (prev status, cur status, camera number) # to get if it has 
    possible_transitions ={}
    pt_start ={}
    pt_end = {}
    for framenumber in range(1, MAXITER): # run through all gt files to find it at that framenumber. 
        print("[INFO] framenumber: {}, STATUS {}, MAXITER {}".format(framenumber, STATUS, MAXITER))
        items = get_candidates(framenumber, allfiles, ids) # (camid, sector, number of vehicles in the view, location of the target vehicle in the view) #* all instance seen across all cameras
        tvcandidates = []

        if STATUS[1] == "NF":
            for item in items: #* instances of the same TV thats seen at "framenumber"
                if item[3]:
                    tvcandidates.append(item)
            print(tvcandidates)
            if tvcandidates: #* found TV in one of tvcandidates
                STATUS = ("NF", "TRK", tvcandidates[0][0])
            else:
                print("[INFO] Not seen yet")

        elif STATUS[1] == "TRK":
            for item in items: #* instances of the same TV thats seen at "framenumber"
                if item[3]:
                    tvcandidates.append(item)
            print(tvcandidates)
            if tvcandidates and checkifsame(STATUS[2], tvcandidates): #* TV is seen from the same camera
                continue
            elif tvcandidates and not checkifsame(STATUS[2], tvcandidates): #* TV is seen from a different camera
                STATUS = ("TRK", "TRK", tvcandidates[0][0])
                print("[INFO] changing camera to {}".format(tvcandidates[0][0]))
            elif not tvcandidates: #* None of the cameras see TV
                STATUS = ("TRK", "TRS", STATUS[2])
                possible_transitions = searchnextcamcandidates(STATUS[2])
                for key, value in possible_transitions.items():
                    pt_start[key] = value[0] + framenumber
                    pt_end[key] = value[1] + framenumber
                print(pt_start, pt_end)
        elif STATUS[1] == "TRS":
            if not possible_transitions:
                STATUS = ("TRS", "L", STATUS[2])
                continue
            for k, v in possible_transitions.items():
                print(framenumber, pt_start[k] , pt_end[k])
                if pt_start[k] <= framenumber and pt_end[k] > framenumber:
                    existence = compare_both(k, items)
                    if existence!=-1:
                        STATUS = ("TRS", "TRK", existence)
                        possible_transitions={}
                elif pt_start[k]  > framenumber and pt_end[k] > framenumber:
                    print("Lets wait")
                elif pt_start[k] < framenumber and pt_end[k] < framenumber:
                    possible_transitions.pop(k)
                    pt_start.pop(k)
                    pt_end.pop(k)
                    print("poped: {}".format(k))
                    break
                
            
        elif STATUS[1] == "L":  
            for cam in all_cameras: #* FIXMI FIRST do BF until you find it 
                if checkifsame(cam, items):
                    STATUS = ("NF", "TRK", cam)
                    print("[INFO] Found at {}".format(cam))
                    break
        
#!10/07 TODO
# -> 이것부터 해보자. spatio-temporal 연관성 ---
# ---> 1) camera 단위로 + Sector 단위로 ---
# ---> 그리고 그리기 
# SUM REID 개수 샐것. 
# accuracy metric 넣을 것. 

# BF 구현                