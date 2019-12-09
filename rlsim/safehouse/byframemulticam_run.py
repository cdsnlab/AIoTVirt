#!/usr/bin/env python

# this program rewards system on per frame. 
# uses 4 cameras to work with.


import logging
import logging.handlers
import gym
import sys
import time
import random
#import camera
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
        default=0.99,
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
        default=4,
        help='number of cameras in sim (def: 2).'
    )
    argparser.add_argument(
        '--totit',
        metavar = 't',
        default=5,
        help='number of total iterations (def: 5).'
    )

    args = argparser.parse_args()

    env = chamEnv(float(args.alp), float(args.gam), float(args.eps), float(args.weight)) # alpha, gamma, epsilon, gpeeweight
    draw = drawgraph()
    #scheme = "eg"
    scheme = args.scheme
    fn_header = "framebase_"
    iteration = 0
    totaliteration = int(args.totit)

    gp_dict = {}
    ee_dict = {}
    svdict = {}

    start_time = time.time()

    dirn = str(args.alp)+str(args.gam)+str(args.eps)+str(args.weight)
    if not os.path.exists(dirn+fn_header):
        os.makedirs(dirn+fn_header)
    f_rew = dirn+fn_header+'/'+scheme+'_finalrewards.csv'
    f_gp = dirn+fn_header+'/'+scheme+'_finalgp.csv'
    f_ee = dirn+fn_header+'/'+scheme+'_finalee.csv'
    f_qtable = dirn+fn_header+'/'+scheme+'_finalqtable.csv'

    env.setscheme(scheme)
    slacknoti("[MEWTWO] spencer starting simulation with " + fn_header)
    while iteration < totaliteration: 
        agent=[]
        slackstring = "[MEWTWO] spencer running " + str (iteration) +"/"+str(totaliteration) + " on cuda 0, 1"
        print (slackstring)
        slacknoti(slackstring)     
        # this does not reset qtable!
        #agent = [camera.cam('01'), camera.cam('02')] # this needs to be running in parallel... or it goes serial.
        for i in range(int(args.numcam)):
            agent.append(camera4.cam(str(i)))

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
        fnqvalue =dirn+fn_header+'/'+ scheme+"_qvalue"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fnstate =dirn+fn_header+'/'+ scheme+"_state"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fnact =dirn+fn_header+'/'+ scheme+"_action"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fncumgp =dirn+fn_header+'/'+ scheme+"_cumgp"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fncumee =dirn+fn_header+'/'+ scheme+"_cumee"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"
        fnwrong =dirn+fn_header+'/'+ scheme+"_wrong"+"_"+str(iteration)+"_"+str(totaliteration)+".csv"

        threads = []
        cidx={}
        def run_thr(target, id):
            res = target[id].procframe(target[id].id,0)
            #cidx.append(res)
            cidx[id] = res

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
#            c2 = agent[1].procframe(agent[1].id, i, env.curaction)
            #print(cidx)
            #env.translatestates(cidx)
            env.translatestatedict(cidx)
            
            print("cstate: ", cidx)
            if "1" not in env.perceivedstatus: # currently we are not seeing anything in any of the cameras. 
                # if currently perceived status does not contain any 1
                psh = env.getstatushistory(int(args.numcam))
                #print("psh: ", psh)
                # if psh is all zeros --> not seen from the camera network. save cstate to be [curloc, prevloc, timer]
                pshcount = psh.count('0')
                if pshcount == int(args.numcam): # never seen in ay of the cam.
                    env.cstate=[psh,psh,0]
                else: #previously seen in some cam.
                    # findout which cam.
                    sind = psh.index(max(psh))
                    env.cstate = ["".join(env.createcombinationexact(int(args.numcam),0)), psh, agent[sind].void]
            else: # currently we are seeing one.
                env.cstate=[env.perceivedstatus, "".join(env.createcombinationexact(int(args.numcam),0)),0]


            '''
            # 여기서 FF인 경우에 이전에 어느 카메라에 있었는지 알수 있어야 함. 
            if not c1 and not c2:
                psh = env.getstatushistory() # prvious camera loc
                if psh == "01":
                    env.cstate = [None, "01", agent[0].void] # tuple3--> 몇 초
                elif psh == "10":
                    env.cstate = [None, "10", agent[1].void]
                else: # when None is returned, it means target has just entered the camera network but not seen. 
                    env.cstate = [None, "00", 0]
                # state -> [curloc, preloc, time]]
            elif c1 and not c2: # IN CAM1
                env.cstate = ["01", None, None] # 여기도 duration으로 설명해야 될듯..
                
            elif not c1 and c2: # IN CAM2
                env.cstate = ["10", None, None]
                
            else: # OVERLAP
                print("no need to discuss yet. ")
                break
            '''
            #print(env.cstate)
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

        iteration += 1
        agent[0].cap.release()
        agent[1].cap.release()

        #clear for next round
        env.statushistory=[]
        env.qvaluecount={}
        env.cumee =0
        env.maxcumee=0
        env.cumgp =0
        
        round_time = time.time()
        print("round took: ", round_time-start_time)

    draw.singlegraphfromdict(gp_dict, dirn+fn_header, "# iteration", "goodput (%)", "gp frame based")
    draw.singlegraphfromdict(ee_dict, dirn+fn_header, "# iteration", "energy usage (%)", "ee frame based")

    end_time = time.time()
    print("sim took: ", end_time-start_time)
    slacknoti("[MEWTWO] spencer done using")

def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

if __name__ == '__main__':
    play()
