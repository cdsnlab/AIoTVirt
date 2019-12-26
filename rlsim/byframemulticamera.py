#!/usr/bin/env python

# this program rewards system on per frame. 
# uses multiple cameras to work with.


import logging
import logging.handlers
import gym
import sys
import time
import random
import cameramulti
import cv2
import _pickle as pickle
from env_multi import chamEnv
from draw import drawgraph
import csv
import requests
import json
import argparse
import configparser
import os
import threading

def play():

    argparser = argparse.ArgumentParser(
        description="wheheh")
    argparser.add_argument(
        '--alp',
        metavar = 'a',
        default = 0.9,
        help='alpha value (def: 0.9)'
    )
    argparser.add_argument(
        '--gam',
        metavar = 'g',
        default=0.9,
        help='gamma value (def:0.9)'
    )
    argparser.add_argument(
        '--eps',
        metavar = 'e',
        default=0.1,
        help='eps value (def:0.1)'
    )
    argparser.add_argument(
        '--weight',
        metavar = 'w',
        default=0.90,
        help='weight btw gp, ee (def: 0.99). if large gp>ee priority.'
    )
    argparser.add_argument(
        '--scheme',
        metavar = 's',
        default='rn',
        help='scheme (def: random noise) options: rn, eg.'
    )
    argparser.add_argument(
        '--numcam',
        metavar = 'n',
        default=6,
        help='number of cameras in sim (def: 2).'
    )
    argparser.add_argument(
        '--totit',
        metavar = 't',
        default=20,
        help='number of total iterations (def: 20).'
    )
    argparser.add_argument(
        '--addarg',
        default="setname",
        help='additional comments for naming (def: 5).'
    )
    argparser.add_argument(
        '--like',
        default="0.3",
        help='likelihood of the dataset (def: 0.3).'
    )

    args = argparser.parse_args()

    env = chamEnv(float(args.alp), float(args.gam), float(args.eps), float(args.weight)) # alpha, gamma, epsilon, gpeeweight
    draw = drawgraph()
    #scheme = "eg"
    scheme = args.scheme
    fn_header = "prop_"
    iteration = 0
    totaliteration = int(args.totit)

    gp_dict = {}
    ee_dict = {}
    svdict = {}
    
    tmpTP = 0
    tmpFP = 0
    tmpTN = 0
    tmpFN = 0
    cumacc = {}
    cumprecision = {}
    cumrecall = {}
    precision = 0
    recall = 0
    accuracy =0 

    start_time = time.time()

    dirn = str(args.like)+"_try"+str(args.addarg)+"_"+str(args.alp)+"_"+str(args.weight)
    if not os.path.exists(fn_header+dirn):
        os.makedirs(fn_header+dirn)
    f_rew = fn_header+dirn+'/'+scheme+'_finalrewards.csv'
    f_gp = fn_header+dirn+'/'+scheme+'_finalgp.csv'
    f_acc = fn_header+dirn+'/'+scheme+'_finalacc.csv'
    f_pre = fn_header+dirn+'/'+scheme+'_finalpre.csv'
    f_rec = fn_header+dirn+'/'+scheme+'_finalrec.csv'
    f_ee = fn_header+dirn+'/'+scheme+'_finalee.csv'
    f_qtable = fn_header+dirn+'/'+scheme+'_finalqtable.csv'

    env.setscheme(scheme)
    slacknoti("[MEWTWO] spencer starting simulation with " + fn_header)
    while iteration < totaliteration: 
        agent=[]
        slackstring = "[MEWTWO] spencer running " + str (iteration) +"/"+str(totaliteration) + " on cuda 0, 1"
        #print (slackstring)
        slacknoti(slackstring)     
        # this does not reset qtable!
        #agent = [camera.cam('01'), camera.cam('02')] # this needs to be running in parallel... or it goes serial.
        for i in range(int(args.numcam)):
            agent.append(cameramulti.cam(str(i), iteration, args.like))

        #agent = [camera.cam('01'), camera.cam('02'), camera.cam('03'), camera.cam('04')] # this needs to be running in parallel... or it goes serial.
        count = 0
        reward = 0
        p = int(args.numcam) # total number of cams 
        e = 2 # number of simultaneous frames to get from.
        if iteration == 0:
            env.setcamnum(len(agent))
            env.reset(p,e)
        else: 
            print ("not resetting")
            time.sleep(1)

        # create csv file names for evaluation
        fnqvalue =fn_header+dirn+'/'+ scheme+"_qvalue"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fnstate =fn_header+dirn+'/'+ scheme+"_state"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fnact =fn_header+dirn+'/'+ scheme+"_action"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fncumgp =fn_header+dirn+'/'+ scheme+"_cumgp"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fncumee =fn_header+dirn+'/'+ scheme+"_cumee"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fnwrong =fn_header+dirn+'/'+ scheme+"_wrong"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"

        threads = []
        cidx={}
        lpos={}
        def run_thr(target, id):
            res, lastseen = target[id].procframe(target[id].id,0)
            #cidx.append(res)
            cidx[id] = res
            lpos[id] = lastseen

        for k in range (0,int(agent[0].cap.get(cv2.CAP_PROP_FRAME_COUNT)-2)):
            print("current frame number: ", count)

            # 원래는 state -> action -> reward 순서로 가야됨. 
            # 1) state
            for i in range(int(args.numcam)):
                #print(agent[j].id)
                #th.append(threading.Thread(target=c[i].procframe, args=(c[i].id, 0)))
                threads.append(threading.Thread(target=run_thr, args=(agent, i)))
                threads[i].start()
            for i in threads:
                i.join()

            threads=[]
            #cidx.append(agent[j].procframe(agent[j].id, k))
            #c1 = agent[0].procframe(agent[0].id, i, env.curaction) # operation 혹은 action 이 필요한듯?
            #c2 = agent[1].procframe(agent[1].id, i, env.curaction)
            #print(cidx)
            env.translatestatedict(cidx)
            env.translatelposdict(lpos) 
            print("current state (cidx): ", cidx)
            if "1" not in env.perceivedstatus: # currently we are not seeing anything in any of the cameras. 
                # if currently perceived status does not contain any 1
                psh = env.getstatushistory(int(args.numcam))
                #print("psh: ", psh)
                # if psh is all zeros --> not seen from the camera network.
                pshcount = psh.count('0')
                if pshcount == int(args.numcam): # never seen in any of the cam.
                    env.cstate=[psh, psh, 0, "xx"] # [curloc, prevloc, timer, locinpixel] locinpixel --> 9x9 matrix of the frame
                    # env.cstate=[psh,psh,0] # [curloc, prevloc, timer]
                else: #previously seen in some cam. findout which cam.
                    sind = psh.index(max(psh))
                    env.cstate = ["".join(env.createcombinationexact(int(args.numcam),0)), psh, agent[sind].void, agent[sind].lpos] # 4state
                    # env.cstate = ["".join(env.createcombinationexact(int(args.numcam),0)), psh, agent[sind].void] 
            else: # currently we are seeing one.
                psh = env.getstatushistory(int(args.numcam))
                sind = psh.index(max(psh))
                env.cstate=[env.perceivedstatus, "".join(env.createcombinationexact(int(args.numcam),0)),0, agent[sind].lpos] #
                # env.cstate=[env.perceivedstatus, "".join(env.createcombinationexact(int(args.numcam),0)),0]

            print("current env.cstate: ", env.cstate)
            env.writecumlativestates(count)

            # 2) action 
            env.chooseaction(env.scheme, count)
            env.writeaction(count, env.action)

            # 3) reward
            reward = env.step(env.action, count) # 여기서 input으로 previous reward 계산 필요.
            #new_state, reward, done, _ = env.step(act) #--> 근데 여기에 return되는게 new_state가 나올수가 없는데 ㅠㅠ 

            env.writecumlativerewards(count, reward) 
            count += 1
            cidx={}
            #print(reward)
            #print("rewards up to now: ", env.sumrewards())
        
        #iteration reports
        with open(fnact, 'w') as fact:
            writer = csv.writer(fact)
            for key, value in env.cumlativeactions.items():
                writer.writerow([key,value])
        with open(fnstate,'w') as wile:
            writer = csv.writer(wile)
            for key, value in env.cumlativestates.items():
                writer.writerow([key,value])
        with open(fnqvalue, 'w') as tile:
            writer = csv.writer(tile)
            for key, value in env.qvaluecount.items():
                writer.writerow([key,value])
        with open(fncumgp, 'w') as qile:
            writer = csv.writer(qile)
            for key, value in env.cumgpdict.items():
                writer.writerow([key,value])
        with open(fncumee, 'w') as cile:
            writer = csv.writer(cile)
            for key, value in env.cumeedict.items():
                writer.writerow([key,value])
        with open(fnwrong, 'w') as aile:
            writer = csv.writer(aile)
            for key, value in env.wrong.items():
                writer.writerow([key,value])
        
        sv = env.sumrewards()
        svdict[iteration] = sv 
        with open (f_rew, 'w') as take:
            writer = csv.writer(take)
            for key, value in svdict.items():
                writer.writerow([key, value])
        # gp
        gp_dict[iteration] = env.cumgp / count * 100
        with open (f_gp, 'w') as make:
            writer = csv.writer(make)
            for key, value in gp_dict.items():
                writer.writerow([key, value])
        # TP, TN, FP, FN 형태로 바꿔도 상관 없는지 확인하고 대체할 것. TN과 FN을 파악할 방법이 있나?
        
        tmpTP = env.TP
        tmpFP = env.FP        
        tmpTN = env.TN
        tmpFN = env.FN
        precision = (tmpTP) / (tmpTP + tmpFP) *100
        recall = (tmpTP) / (tmpTP+tmpFN) * 100
        accuracy = (tmpTP + tmpTN) / (tmpTP+tmpFP+tmpTN+tmpFN) * 100
        #result += ((tmpTPTN) / (tmpTPTN+tmpFPFN))*100
        print("tpfptnfn합: ", tmpTP+tmpFP+tmpTN+tmpFN) # 이거 누적되면 곤란한데..
        print("count: ", count)
        print("accuracy= ", accuracy)
        print("recall= ", recall)
        print("precision= ", precision)
        cumacc[iteration] = accuracy 
        cumrecall[iteration] = recall
        cumprecision[iteration] = precision

        with open (f_acc, 'w') as make:
            writer = csv.writer(make)
            for key, value in cumacc.items():
                writer.writerow([key, value])
        
        with open (f_rec, 'w') as saake:
            writer = csv.writer(saake)
            for key, value in cumrecall.items():
                writer.writerow([key, value])
        
        with open (f_pre, 'w') as amake:
            writer = csv.writer(amake)
            for key, value in cumprecision.items():
                writer.writerow([key, value])
        # ee
        ee_dict[iteration] = env.cumee / env.maxcumee * 100
        with open (f_ee, 'w') as wake:
            writer = csv.writer(wake)
            for key, value in ee_dict.items():
                writer.writerow([key, value])
        # print q table
        with open(f_qtable, 'w') as pake:
            writer = csv.writer(pake)
            for key, value in env.qkey.items():
                writer.writerow([key, value])


        #clear for next round
        env.statushistory=[]
        env.qvaluecount={}
        env.cumee =0
        env.maxcumee=0
        env.cumgp =0

        env.TP = 0
        env.FP = 0
        env.TN = 0
        env.FN = 0
        iteration += 1
        
        round_time = time.time()
        print("round took (seconds): ", round_time-start_time)
    
    draw.singlegraphfromdict(cumacc, fn_header+dirn, "# iteration", "accuracy (%)", "accuracy frame based")
    draw.singlegraphfromdict(cumrecall, fn_header+dirn, "# iteration", "recall (%)", "recall frame based")
    draw.singlegraphfromdict(cumprecision, fn_header+dirn, "# iteration", "precision (%)", "precision frame based")
    draw.singlegraphfromdict(gp_dict, fn_header+dirn, "# iteration", "goodput (%)", "gp frame based")
    draw.singlegraphfromdict(ee_dict, fn_header+dirn, "# iteration", "energy usage (%)", "ee frame based")

    end_time = time.time()
    print("sim took (seconds): ", end_time-start_time)
    slacknoti("[MEWTWO] spencer done using")

def slacknoti(contentstr):
    
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

if __name__ == '__main__':
    play()