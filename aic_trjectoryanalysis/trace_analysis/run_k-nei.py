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
idlistdb = db['idlists']
stalldb = db['st_all']
# stdb = db['spatio_temporal'] #* does not contain immidiate transitions between cams when it ends with a trace
gtdb = db['gt']

rtresdb = db['runtime_results'] #* saves [Scheme, Framenumber, IDs, Sector, Reid counts per frame, Cams looked at per frame]

# S01, S02 has only 4, 5 camera involved.
# S03, S04, S05 has 6, 23, 19 cameras involved.
SECTORTYPE = "S05" 


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

def get_number_of_obj_in_same_cam(items, target_cam):
    other_vehicle_cnt=0
    for item in items:
        if target_cam == item[2]:
            other_vehicle_cnt += 1
    return other_vehicle_cnt

def target_generator(sector_type, order, bucket, iteration): #* Sector, ORDER: seq/rand, REQ: number of concurrent UIDs, ITER: how many sets do you want
    lookup = {"sectors": sector_type}
    doc=list(idlistdb.find(lookup, {"_id":0, "ids":1, "sectors":1}))
    #print(doc)
    cnt =0
    runable = []
    if order == "s":
        for i in range(iteration):
            tmp  = []
            for j in range(bucket):
                #print(doc[0]['ids'][cnt])
                tmp.append(doc[0]['ids'][cnt])
                cnt+=1
            runable.append(tmp)

    elif order == "r":
        for i in range(iteration):
            tmp = []
            for j in range(bucket):
                tmp.append(doc[0]['ids'][random.randint(1,len(doc[0]['ids']))])
                #tmp.append(random.randint(1, max))
            runable.append(tmp)
    #print (runable)
    return runable

def target_generation_partial(sector_type, order, bucket):
    lookup = {"sectors": sector_type}
    doc=list(idlistdb.find(lookup, {"_id":0, "ids":1, "sectors":1}))
    # print(len(doc[0]['ids']))
    maxiter = (len(doc[0]['ids']) // bucket) * bucket
    # print(maxiter*bucket)
    cnt = 0
    runable = []
    if order == "s":
        while cnt != (len(doc[0]['ids']) // bucket) * bucket: #그 사이에 지나가버릴 수 도 있음. 
            tmp=[]
            for i in range(bucket): #bucket 개수만큼  담고 하나씩 넘어 갈 것. 
                tmp.append(doc[0]['ids'][cnt])
                cnt+=1
            if len(tmp) < bucket:
                continue
            else:
                runable.append(tmp)
            # print(cnt, tmp)
    return runable

def target_generation_all(sector_type, order):
    lookup = {"sectors": sector_type}
    doc=list(idlistdb.find(lookup, {"_id":0, "ids":1, "sectors":1}))
    maxbucket = 10    
    runable = []
    if order == "s":
        for bucket in range(1, maxbucket):
            cnt = 0
            while cnt != (len(doc[0]['ids']) // bucket) * bucket: #그 사이에 지나가버릴 수 도 있음. 
                tmp=[]
                for i in range(bucket): #bucket 개수만큼  담고 하나씩 넘어 갈 것. 
                    tmp.append(doc[0]['ids'][cnt])
                    cnt+=1
                if len(tmp) < bucket:
                    continue
                else:
                    runable.append(tmp)
                # print(cnt, tmp)
    return runable


#* creates k number of target ID's to track. 
#* can choose, sequential vs random

#* Sector number, returns sequential, number of target(s) UIDs to search concurrently, how many iterations
#req = target_generator(SECTORTYPE, "s", 2, 2) #? 테스트용으로 일부 셋트 만드는 것.
#req = target_generation_partial(SECTORTYPE, "s", 5) #? n개 조합, 
req=target_generation_all(SECTORTYPE, "s") #? 전체 시뮬레이션에 필요한 loop개수, 1~10대까지 조합 만들어서 return. ==> 모든 조합은 불가능/불필요
#req = [[1], [2], [3], [4], [5]] # req example
print(req)
for ids in tqdm(req): #* ids is a list
    initids = ids
    
    MAXITER = get_maxiter(ids) #dict
    STATES = defaultdict(lambda: defaultdict(dict))
    items = None
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
        camslookedat =defaultdict(list) 
        reidcnts =defaultdict(int) 
        print("[INFO] framenumber: {}, MAXITER {}".format(framenumber, max(MAXITER.items(), key=operator.itemgetter(1))[1]))
        items = get_items_in_all_views(SECTORTYPE, framenumber)
        #print(items)

        for i in ids:
            print("[INFO] id: {}, STATES {}".format(i, STATES[i]))
            if STATES[i]['cur'] == "NF":
                for item in items:
                    # reidcnts += get_number_of_obj_in_same_cam(items, item[2])
                    reidcnts[i]+=1 # 어짜피 한 item이 해당 framenumber에 있는 하나의 vehicle을 의미함. 
                    if item[2] not in camslookedat[i]:
                        camslookedat[i].append(item[2])
                    if item[3] == i: #애초에 존재하는 물체 모두 return하기 때문에, 같은 id가 없으면 xywh도 없다고 보면 됨.
                        print("Found at {}".format(item[2]))
                        STATES[i]['cur']="TRK"
                        STATES[i]['camera']=item[2]
                        break
                    
            elif STATES[i]['cur'] =="TRK":
                for item in items: 
                    if item[2] == str(STATES[i]['camera']): 
                        STATES[i]['cur']="TRK"
                        print("Still at {}".format(item[2]))
                        reidcnts[i] += get_number_of_obj_in_same_cam(items, item[2])#? return other vehicles seen at cam item[2]. 어짜피 cur cam 만 보기로 했으니 다른 cam은 reid하지 않아도 됨.
                        # if STATES[i]['prev']=="TRK": #! 이거 여기 있을 이유가 없음. 어짜피 cur cam만 보기로 했으니.
                        #     camslookedat[i]=[]  # empty it since its full of trash
                        if item[2] not in camslookedat[i]:
                            camslookedat[i].append(item[2])
                        break
                    else: # 없다면 TRS로 변경.
                        # if STATES[i]['camera'] not in camslookedat[i]: #! 이거 여기 있을 이유가 없음. 어짜피 cur cam만 보기로 했으니.
                        #     camslookedat[i].append(STATES[i]['camera'])
                        STATES[i]['cur']="TRS"
                        STATES[i]['prev']="TRK"
                        STATES[i]['possible_trans'] = search_next_cam_candidates(STATES[i]['camera'], SECTORTYPE) 
                        #? 애초에 DB에 넣을때 start와 end만 넣지, 중간에 보였다가 다시 보이는 케이스가 안 다뤄져있음. 따라서 self transition 추가
                        for key, value in STATES[i]['possible_trans'].items(): 
                            STATES[i]['pt_start'][key] = value[0] + framenumber
                            STATES[i]['pt_end'][key] = value[1] + framenumber

            elif STATES[i]['cur'] =="TRS":
                if not STATES[i]['possible_trans']:
                    STATES[i]['cur']="L"
                    STATES[i]['prev']="TRS"
                    continue
                for k, v in STATES[i]['possible_trans'].items(): # K: CAMERA???, 
                    print(k, STATES[i]['pt_start'][k], STATES[i]['pt_end'][k])
                    if STATES[i]['pt_start'][k] <= framenumber and STATES[i]['pt_end'][k] > framenumber: #*possible cam transition! look at it!
                        reidcnts[i] += get_number_of_obj_in_same_cam(items, k)#* return other vehicles seen at cam item[2]. 어짜피 cur cam 만 보기로 했으니 다른 cam은 reid하지 않아도 됨.
                        if k not in camslookedat[i]: 
                            camslookedat[i].append(k)
                        if compare_both(k, items)!=-1:
                            STATES[i]['cur']="TRK"
                            STATES[i]['prev'] ="TRS"
                            STATES[i]['camera']=compare_both(k, items)
                            print("[INFO] Found at {}".format(compare_both(k, items)))
                            #* clearing possible transitions
                            STATES[i]['possible_trans'] = {}
                            STATES[i]['pt_start'] = {}
                            STATES[i]['pt_end'] = {}
                            break #* found  it, lets go back to tracking
                    elif STATES[i]['pt_start'][k]  > framenumber and STATES[i]['pt_end'][k] > framenumber: #*we haven't reached its time yet. don't look at it yet.
                        print("Lets wait")
                    elif STATES[i]['pt_start'][k] < framenumber and STATES[i]['pt_end'][k] <= framenumber: #* we've reached maximum time!
                        reidcnts[i] += get_number_of_obj_in_same_cam(items, k)#* return other vehicles seen at cam item[2]. 어짜피 cur cam 만 보기로 했으니 다른 cam은 reid하지 않아도 됨.
                        if k not in camslookedat[i]: 
                            camslookedat[i].append(k)
                        STATES[i]['possible_trans'].pop(k)
                        STATES[i]['pt_start'].pop(k)
                        STATES[i]['pt_end'].pop(k)
                        print("poped: {}".format(k))
                        break
            elif STATES[i]['cur'] =="L":
                for item in items:
                    reidcnts[i]+=1 # 어짜피 한 item이 해당 framenumber에 있는 하나의 vehicle을 의미함.
                    if item[2] not in camslookedat: 
                        camslookedat[i].append(item[2])
                    if item[3] == i: # 해당 하는 ID가 있는지 확인.
                        print("Found at {}".format(item[2]))
                        STATES[i]['cur']="TRK"
                        STATES[i]['prev']="L"
                        STATES[i]['camera']=item[2]
                        print("[INFO] Found at {}".format(item[2]))
                        break
                
        allreidcnts=0
        allcamslookedat = []
        for i in ids: #! this doesn't ensure duplicated camera views for mutli target tracking
            allreidcnts += reidcnts[i]
            allcamslookedat.append(camslookedat[i])
        inputrow = {"scheme": "k-nei", "framenumber": framenumber, "ids": ids, "sector": SECTORTYPE, "initids": initids, "reidcnts": allreidcnts, "camslookedat": allcamslookedat } 
        print(inputrow)
        # rtresdb.insert_one(inputrow)
        #* pop it if its the end of id x's last seen time
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
# DB 수정. ---
# multi target version으로 수정 ---

#! 10/11 TODO
# SUM REID 개수 샐것. 
# accuracy metric 넣을 것. ()
# BF 구현