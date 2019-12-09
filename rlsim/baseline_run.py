#!/usr/bin/env python

#this program outputs 

import logging
import logging.handlers
import gym
import sys
import time
import random
import camerabaseline
import cv2
import _pickle as pickle
from draw import drawgraph
import csv
import requests
import json
import argparse
import configparser
import os
import threading
import operator

cumlativegp={} # cumlative gp

def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

def play():

    argparser = argparse.ArgumentParser(
        description="wheheh")
    argparser.add_argument(
        '--numcam',
        metavar = 'n',
        default=4,
        help='number of cameras in sim (def: 2).'
    )
    argparser.add_argument(
        '--totit',
        metavar = 't',
        default=20,
        help='number of total iterations (def: 5).'
    )

    draw = drawgraph()

    args = argparser.parse_args()
    agent =[]
    threads =[]
    cidx={}
    cumact={}
    count = 0
    waitframe = 0
    last = 0
    tmpTP = 0
    tmpTN = 0
    tmpFP = 0
    tmpFN = 0
    tmpee=0
    result=0
    cumgp ={}
    cumee={}
    final = 0

    start_time = time.time()
    fn_header = "baseline_07"
    iteration = 0
    totaliteration = int(args.totit)
    dirn = ""
    if not os.path.exists(fn_header+dirn):
        os.makedirs(fn_header+dirn)
    f_gp = fn_header+dirn+'/'+'_finalgp.csv'
    f_ee = fn_header+dirn+'/'+'_finalee.csv'

    slacknoti("[MEWTWO] spencer starting simulation with " + fn_header)
    while iteration < totaliteration: 
        newact = "find"

        agent=[]
        count=0
        waitframe=0
        slackstring = "[MEWTWO] spencer running " + str (iteration) +"/"+str(totaliteration) + " on cuda 0, 1"
        slacknoti(slackstring)     

        for i in range(int(args.numcam)):
            agent.append(camerabaseline.cam(str(i), iteration))

        def run_thr(target, id, newact):
            res = target[id].procframe(target[id].id,newact)
            #cidx.append(res)
            cidx[id] = res

        for k in range (0, int(agent[0].cap.get(cv2.CAP_PROP_FRAME_COUNT)-2)):
            #print ("current frame number", count)
            cumact[count] = newact
            for i in range(int(args.numcam)):
                threads.append(threading.Thread(target=run_thr, args=(agent, i, newact)))
                threads[i].start()

            for i in threads:
                i.join()
            threads=[]

            newact = changeact(newact, cidx)
            #print("[main] newact: ", newact)
            if str(newact)[-1:]=="s":
                #print("[main] :", str(newact)[-1:])
                waitframe = int(newact[0]) * 15
                #print("[main] wait time: ", waitframe)
                newact = "SHUT"

            if waitframe == 0:
                if newact != "find" and newact == "SHUT": # in cam
                #    pass
                #elif newact == "SHUT": # not in any cam
                    last = lookuplatest(cumact, count)
                    #print("LAST: ", last)
                    newact = int(last) + 1

                # transit to next one (how do we know which one was last)
            

            else: 
                if newact=="SHUT":
                    waitframe -= 1
                    #print("shuttingup")
            #print(waitframe)

            if newact == "find" or newact == "SHUT":
                tmpee +=0
            else:
                tmpee+=1
                       
            count+=1
        round_time = time.time()
        print("round took: ", round_time-start_time)

        for i in range(len(agent)):
            #print(agent[i].TP, agent[i].TN, agent[i].FP, agent[i].FN)
            tmpTP = agent[i].TP
            tmpTN = agent[i].TN
            tmpFP = agent[i].FP
            tmpFN = agent[i].FN
            result += ((tmpTP+tmpTN) / (tmpTP+tmpTN+tmpFP+tmpFN))*100
            print(tmpTP, tmpTN, tmpFP, tmpFN)
            print(result)
        #print ("result: ", (TP+FN) / (TP+TN+FP+FN))
        cumgp[iteration] = result / int(args.numcam)
        
        with open(f_gp, 'w') as pake:
            writer = csv.writer(pake)
            for key, value in cumgp.items():
                writer.writerow([key, value])

        cumee[iteration] = (tmpee / (count * int(args.numcam))) * 100
        with open(f_ee, 'w') as pake:
            writer = csv.writer(pake)
            for key, value in cumee.items():
                writer.writerow([key, value])

        tmpTP = 0
        tmpTN = 0
        tmpFP = 0
        tmpFN = 0
        tmpee=0
        result =0
        iteration+=1

    draw.singlegraphfromdict(cumgp, fn_header+dirn, "# iteration", "goodput (%)", "gp frame based")
    draw.singlegraphfromdict(cumee, fn_header+dirn, "# iteration", "energy usage (%)", "ee frame based")
    ###swtesting code
    for key, value in cumgp.items():
        final += cumgp[key]
    final = final / (int(iteration))
    print ("average: ", final)
    ###
    slacknoti("[MEWTWO] spencer done using")

def lookuplatest(cumact, count):
    # search for the latest position of camera 
    #sorted_d = dict(sorted(cumact.items(), key=operator.itemgetter(0), reverse=True))
    #print(sorted_d)
    for i in range(count, 0, -1):
        if len(str(cumact[i]))==1:
            #print (int(cumact[i]))
            return int(cumact[i])

def changeact(newact, cidx):
    if newact=="find":
        for key, value in cidx.items():
            if value==True:
                return key
            else:
                return newact

    elif newact == "SHUT": # suppress other cameras as target is in btw blind spot
        #print("is suppressed")
        return newact
    else: 
        for key, value in cidx.items():
            if value!=True and  value !=False:
                return str(cidx[key])+"s"
        
        #print("in cam state")

        return newact

def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})



if __name__ == '__main__':
    play()