'''
[DONE]
Functions needed to run run_*.py is in this file
'''
from pymongo import MongoClient
from collections import defaultdict
# from functionalities import get_items_in_all_views, search_next_cam_candidates, compare_both, get_maxiter, get_number_of_obj_in_same_cam, target_generation_all, target_generation_partial, target_generator
import random
import operator

client = MongoClient('localhost', 27017)
db = client['aic_mtmc']
iddb = db['uid_traces']
idlistdb = db['idlists']
stalldb = db['st_all']
# stdb = db['spatio_temporal'] #* does not contain immidiate transitions between cams when it ends with a trace
gtdb = db['gt']
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
