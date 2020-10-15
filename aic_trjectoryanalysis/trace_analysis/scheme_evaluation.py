'''
[Prog]
INPUT: rtresdb db
OUTPUT: comaprison of 1) accuracy, 2) precision, 3) recall, 4) 'additional' reid-cnts, 5) number of 'additional' cameras viewed. of each [req] set

- again, generate all set of reqs for querying mongodb -> [req] is going to be used as the query key
- create another db to save accuracy, precision, recall, etc
loop on every frame
'''
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
gtdb = db['gt_s05']
# gtdb = db['gt']

SECTORTYPE = "S05" 
# rtresdb = db['runtime_results'] #* saves [Scheme, Framenumber, IDs, Sector, Reid counts per frame, Cams looked at per frame]
# req= target_generator(SECTORTYPE, "s", 2, 2) #! use with runtime_results.db
rtresdb = db['rt_results']
req = target_generation_partial (SECTORTYPE, "s", 3) #use this with rt_results.db
# rrtresdb = db['rt_results'] #! this is the full set
#req=target_generation_all(SECTORTYPE, "s") #! use with rt_results.db
req=[[335,341]]

def exists_in_any_list(gt, candidate):
    # assume scheme, framenumber, id is the same. only difference is the camera lists.
    for g in gt:
        for c in candidate:
            if g['camera'] == c:
                return True
    # gt_results[idx][i]['camera'] in sch_results[scheme]['camslookedat'][i]

for ids in tqdm(req): #* ids is a list
    initids = ids.copy()
    
    MAXITER = get_maxiter(ids) #dict
    STATES = defaultdict(dict)
    evaluated = defaultdict(lambda:[0, 0, 0, 0]) #* TP, TN, FN, FP
    vehicle_cnt = defaultdict(lambda: {'gt':0, 'reided':0}) #* counts gt vehicle count VS amount of vehicles to be reid per scheme
    camera_cnt = defaultdict(lambda:0) #* counts extra number of cameras looked at each time step.
    result = defaultdict(lambda:defaultdict(lambda: {'accuracy':0, 'precision':0, 'recall':0}))
    for framenumber in range(1, max(MAXITER.items(), key=operator.itemgetter(1))[1]): # run through all gt files to find it at that framenumber. 
        print("[INFO] framenumber: {}, MAXITER {}".format(framenumber, max(MAXITER.items(), key=operator.itemgetter(1))[1]))

        lookup = {"sector": SECTORTYPE, "framenumber":framenumber, "initids": initids} 
        sch_results={}
        for sr in list(rtresdb.find(lookup, {"_id":0, "scheme": 1, "initids": initids, "ids":1 , "framenumber": 1, "cur_states": 1, "reidcnts":1, "camslookedat":1})):
            sch_results[sr['scheme']] = sr
            ids = sr['ids'] # 이거 id가 줄어들 수도 있음. 따라서 sr에서 읽어온 ids를 기반으로 해야지 맞음.
        gt_results = {}
        for idx in ids:  
            gtlookup = {"sector": SECTORTYPE, "framenumber": framenumber, "id": idx} # should return all id, cameras seen at each framenumber
            gt_results[idx]=(list(gtdb.find(gtlookup, {"_id": 0, "framenumber": 1, "camera":1, "id":1})) )
        print("sch_results", sch_results)
        print("gt_results", gt_results)
        
        #TODO 1) ACCURACY
        for i, idx in enumerate(ids):#! 나눠서 기록하고 마지막에 모든 TP, TN, FP, FN 합쳐서? -> 따로해서 합쳐도 상관 없나?            
            if gt_results[idx]: #* gt 에 id가 있다.
                for scheme in sch_results:#* 각 scheme 별로
                    vehicle_cnt[scheme]['gt']+=1
                    if sch_results[scheme]['cur_states'][i] != "NF": #? should ignore if cur_state is in NF
                        camera_cnt[scheme] += len(sch_results[scheme]['camslookedat'][i])
                        vehicle_cnt[scheme]['reided']+=sch_results[scheme]['reidcnts']//len(ids) 
                        if exists_in_any_list(gt_results[idx], sch_results[scheme]['camslookedat'][i]): # TP
                            evaluated[scheme][0]+=1
                        else: #FN
                            evaluated[scheme][2]+=1
                            
            elif not gt_results[idx]: #* gt 에 id가 없다.
                for scheme in sch_results:#* 각 scheme 별로
                    if sch_results[scheme]['cur_states'][i] != "NF": #? should ignore if cur_state is in NF
                        vehicle_cnt[scheme]['reided']+=sch_results[scheme]['reidcnts']//len(ids)
                        # print(idx, scheme, sch_results[scheme]['camslookedat'][i])
                        if sch_results[scheme]['camslookedat'][i]: # FP #! this is controversy
                            camera_cnt[scheme] += len(sch_results[scheme]['camslookedat'][i]) #! 이거 id가 달라도 L 상태이면 보는 캠이 겹칠 수가 있음.
                            evaluated[scheme][3]+=1                             
                        elif not sch_results[scheme]['camslookedat'][i]: #TN... 
                            evaluated[scheme][1]+=1

    
    for scheme in sch_results.keys():
        #! are the metrics okay? double check. 
        result[str(initids)][scheme]['accuracy'] += (evaluated[scheme][0]+evaluated[scheme][1]) / (evaluated[scheme][0]+evaluated[scheme][1]+evaluated[scheme][2]+evaluated[scheme][3])
        result[str(initids)][scheme]['precision'] += evaluated[scheme][0] / (evaluated[scheme][3] + evaluated[scheme][0]) #TP / (FP + TP)
        result[str(initids)][scheme]['recall'] += evaluated[scheme][0] / (evaluated[scheme][2] + evaluated[scheme][0]) #TP / (FN + TP)
    print(result)
    print(vehicle_cnt)
    print(camera_cnt)


#! 2020 10 14
# add re-id counts, extra cams looked at. 

#! 2020 10 15 
# write it a db. 