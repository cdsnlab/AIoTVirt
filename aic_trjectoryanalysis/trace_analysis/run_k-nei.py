"""
[Prog]
INPUT: gt db
OUTPUT: number of frames processed, number of reid-ed objects (vehicles) per frame -> to db
This program finds the number of cameras which needs to be turned on to track a vehicle "while using k-nei scheme". GIVEN that all re-id is PERFECT.
#?Each simulation ends when the vehicle is no long in the network. Should we give it additonal time (e.g. 100 frames = 10seconds) ???
"""

from tqdm import tqdm
from pymongo import MongoClient
from collections import defaultdict
from functionalities import get_items_in_all_views, search_next_cam_candidates, compare_both, get_maxiter, get_number_of_obj_in_same_cam, target_generation_all, target_generation_partial, target_generator
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
    #STATES = defaultdict(lambda: defaultdict(dict))
    STATES = defaultdict(dict)
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
        inputrow = {"scheme": "k-nei", "framenumber": framenumber, "ids": ids, "sector": SECTORTYPE, "initids": initids, 'STATES': STATES, "reidcnts": allreidcnts, "camslookedat": allcamslookedat }  
        print(inputrow)
        # rtresdb.insert_one(inputrow)
        #* pop it if its the end of id x's last seen time
        for k, v in MAXITER.items(): # make sure to pop id if it hits any of the maxiter values.
            if v == framenumber:
                print("=============pop {}".format(k))
                del STATES[k]
                ids.remove(k)
     
    