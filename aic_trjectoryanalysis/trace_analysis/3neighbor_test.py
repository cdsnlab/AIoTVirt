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
from collections import defaultdict
import random
import operator

client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
iddb = db['uid_traces']
stdb = db['spatio_temporal']
idlistdb = db['idlists']
stalldb = db['st_all']
gtdb = db['gt']

# S01, S02 has only 4, 5 camera involved.
# S03, S04, S05 has 6, 23, 19 cameras involved.
SECTORTYPE = "S05" 

def get_camid_section(filename):
    camid, section = None, None
    items = filename.split('/')
    for item in items:
        if item.startswith("c0"):
            camid = item
        if item.startswith("S"):
            section = item
    return camid, section

def get_items_in_all_views(sector, framenumber):
    items = []
    lookup = {"sector": sector, "framenumber": framenumber}
    doc=list(gtdb.find(lookup, {'_id': 0, "sector":1, "framenumber":1, "id":1, 'camera':1, 'xmin':1, 'ymin':1, 'width':1, 'height':1}))
    #print(doc)
    for d in doc:
        #allobjects={"camera": d['camera'], "framenumber": framenumber} #* per camera bases.
        #ao=list(gtdb.find(allobjects, {"_id": 0, "camera":1, "sector": 1, "framenumber":1 , "id":1}))
        #print(ao)
        items.append((framenumber, sector, d['camera'], d['id'], [d['xmin'], d['ymin'], d['width'], d['height']] ))
    return items
    

def search_next_cam_candidates(src, src_sector):
    stitems={}
    srclookup={"src": src}
    #srclookup={"src": src, "src_s": src_sector}
    #print(srclookup)
    doc=list(stalldb.find(srclookup, {"_id":0, "src":1, "dst": 1, "temporal":1}))
    #print(doc)
    for d in doc:
        #print(min(d['temporal']), max(d['temporal']))
        stitems[d['dst']] = (min(d['temporal']), max(d['temporal']))
    stitems[src] = (0, 20)
    return stitems

def compare_both(predicted_cam, all_things):
    # print(predicted_cam, all_things)
    # for i in predicted_cam:
    for j in all_things:
        # print(j[2], predicted_cam)
        if j[2]==predicted_cam:
            return predicted_cam
    return -1

def get_maxiter(ids):
    maxlen = 0
    maxvalue={}
    for id in ids:
        dblookup = {"id": str(id)}
        doc=list(iddb.find(dblookup, {"_id":0, "camid":1, "trace":1}))
        for d in doc:
            #print(d['trace'][-1])
            if int(d['trace'][-1]) > maxlen:
                #print(maxlen)
                maxlen = int(d['trace'][-1]) 
        maxvalue[id]=maxlen
        print ("Maximum iteration for {}:{}".format(id, maxlen))
    #print(maxvalue)
    return maxvalue


def target_generator(sector_type, order, req, iteration): #* Sector, ORDER: seq/rand, REQ: number of concurrent UIDs, ITER: how many sets do you want
    lookup = {"sectors": sector_type}
    doc=list(idlistdb.find(lookup, {"_id":0, "ids":1, "sectors":1}))
    #print(doc)
    cnt =0
    runable = []
    if order == "s":
        for i in range(iteration):
            tmp  = []
            for j in range(req):
                #print(doc[0]['ids'][cnt])
                tmp.append(doc[0]['ids'][cnt])
                cnt+=1
            runable.append(tmp)

    elif order == "r":
        for i in range(iteration):
            tmp = []
            for j in range(req):
                tmp.append(doc[0]['ids'][random.randint(1,len(doc[0]['ids']))])
                #tmp.append(random.randint(1, max))
            runable.append(tmp)
    #print (runable)
    return runable

#* creates k number of target ID's to track. 
#* can choose, sequential vs random
#req = [[1], [2], [3], [4], [5]]
#* Sector number, returns sequential, number of target(s) UIDs to search concurrently, how many iterations
req = target_generator(SECTORTYPE, "s", 2, 2)

for ids in tqdm(req): #* ids is a list
    MAXITER = get_maxiter(ids) #dict
    STATES = defaultdict(lambda: defaultdict(dict))
    GSTATUS=defaultdict(list)
    items = None
    STATUS={}
    possible_transitions ={}
    for i in ids: #* initalize states.
        #STATES[i] = ("NF", "NF", -1) # (prev status, cur status, camera number) # to get if it has 
        STATES[i]['prev'] = "NF"
        STATES[i]['cur'] = "NF"
        STATES[i]['camera'] = -1
        STATES[i]['possible_trans'] = {}
        STATES[i]['pt_start'] = {}
        STATES[i]['pt_end'] = {}
    
    #print(max(MAXITER.items(), key=operator.itemgetter(1))[1])
    for framenumber in range(1, max(MAXITER.items(), key=operator.itemgetter(1))[1]): # run through all gt files to find it at that framenumber. 
        print("[INFO] framenumber: {}, MAXITER {}".format(framenumber, max(MAXITER.items(), key=operator.itemgetter(1))[1]))
        items = get_items_in_all_views(SECTORTYPE, framenumber)
        # print(items)
        tvcandidates = []
        
        for i in ids:
            print("[INFO] id: {}, STATES {}".format(i, STATES[i]))
            if STATES[i]['cur'] == "NF":
                for item in items:
                    if item[3] == i: #애초에 존재하는 물체 모두 return하기 때문에, 같은 id가 없으면 xywh도 없다고 보면 됨.
                        print("Found at {}".format(item[2]))
                        STATES[i]['cur']="TRK"
                        STATES[i]['camera']=item[2]
                        break
            elif STATES[i]['cur'] =="TRK":
                for item in items: 
                    if item[2] == str(STATES[i]['camera']): # 이전에 봤던 물건이랑 같이 일치하는지 확인.
                        STATES[i]['cur']="TRK"
                        print("Still at {}".format(item[2]))
                        break
                    else: # 없다면 TRS로 변경.
                        STATES[i]['cur']="TRS"
                        STATES[i]['prev']="TRK"
                        STATES[i]['possible_trans'] = search_next_cam_candidates(STATES[i]['camera'], SECTORTYPE) 
                        for key, value in STATES[i]['possible_trans'].items(): #! 애초에 DB에 넣을때 start와 end만 넣지, 중간에 보였다가 다시 보이는 케이스가 안 다뤄져있음. 따라서 self transition 추가
                            STATES[i]['pt_start'][key] = value[0] + framenumber
                            STATES[i]['pt_end'][key] = value[1] + framenumber

            elif STATES[i]['cur'] =="TRS":
                if not STATES[i]['possible_trans']:
                    STATES[i]['cur']="L"
                    STATES[i]['prev']="TRS"
                    continue
                for k, v in STATES[i]['possible_trans'].items(): #! 여기서 찾을 것은, 카메라임. ID가 아니라
                    # print(k, STATES[i]['pt_start'][k], STATES[i]['pt_end'][k])
                    if STATES[i]['pt_start'][k] <= framenumber and STATES[i]['pt_end'][k] > framenumber:
                        existence = compare_both(k, items)
                        if existence!=-1:
                            STATES[i]['cur']="TRK"
                            STATES[i]['prev'] ="TRS"
                            STATES[i]['camera']=existence
                            print("[INFO] Found at {}".format(existence))
                            #* clearing possible transitions
                            STATES[i]['possible_trans'] = {}
                            STATES[i]['pt_start'] = {}
                            STATES[i]['pt_end'] = {}
                            break #* found  it, lets go back to tracking
                    elif STATES[i]['pt_start'][k]  > framenumber and STATES[i]['pt_end'][k] > framenumber:
                        print("Lets wait")
                    elif STATES[i]['pt_start'][k] < framenumber and STATES[i]['pt_end'][k] < framenumber:
                        STATES[i]['possible_trans'].pop(k)
                        STATES[i]['pt_start'].pop(k)
                        STATES[i]['pt_end'].pop(k)
                        print("poped: {}".format(k))
                        break
            elif STATES[i]['cur'] =="L":
                for item in items:
                    if item[3] == i: # 해당 하는 ID가 있는지 확인.
                        print("Found at {}".format(item[2]))

                        STATES[i]['cur']="TRK"
                        STATES[i]['prev']="L"
                        STATES[i]['camera']=item[2]
                        print("[INFO] Found at {}".format(item[2]))
                        break
                
            
        for k, v in MAXITER.items(): # make sure to pop id if it hits any of the maxiter values.
            if v == framenumber:
                print("=============pop {}".format(k))
                del STATES[k]
                ids.remove(k)
    
#!10/07 TODO
# -> 이것부터 해보자. spatio-temporal 연관성 ---
# ---> 1) camera 단위로 + Sector 단위로 ---
# ---> 그리고 그리기 ---

#! 10/08 TODO
# concurrent UID input로 변경? ---
# ### 각각 ID에 대한 STATUS로 변경. ---
#! 10/09 TODO
# DB 수정. 

#! 10/11 TODO
# SUM REID 개수 샐것. 
# accuracy metric 넣을 것. ()
# BF 구현