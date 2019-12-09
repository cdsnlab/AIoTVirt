#!/usr/bin/env python
import logging
import logging.handlers
import gym
import sys
import time
import random
import camera
import cv2
from env1 import chamEnv

def play():
    env = chamEnv()
    scheme = "random" # hahahazhaha
    #scheme = "eg" # egreedy
    count = 0
    reward = 0
    env.setscheme(scheme)
    agent = [camera.cam('01'), camera.cam('02')] # this needs to be running in parallel... or it goes serial.
    #p = len(agent) # number of cameras in simulation
    p = 2
    e = 2 # number of simultaneous frames to get from.

    # print("cam1: ", agent[0].cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("cam2: ", agent[1].cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    env.reset(p,e)

    for i in range (0,int(agent[0].cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        
        # 원래는 state -> action -> reward 순서로 가야됨. 
        # 1) state
        c1 = agent[0].procframe(agent[0].id, i, env.curaction) # operation 혹은 action 이 필요한듯?
        c2 = agent[1].procframe(agent[1].id, i, env.curaction)
        env.translatestates(c1,c2)

        # 여기서 FF인 경우에 이전에 어느 카메라에 있었는지 알수 있어야 함. 
        if not c1 and not c2:
            psh = env.getstatushistory() # prvious camera loc
            if psh == "01":
                env.cstate = [None, "01", agent[0].void] # 몇 초? 아니면 몇 frame으로 구해야지!
                #print("wwwww", env.deconstructstatefromlist(env.cstate))
                #print("wefwf", env.deconstructstatefromstring("11,124,129"))
            elif psh == "10":
                env.cstate = [None, "10", agent[1].void]
            else: # when None is returned, it means target has just entered the camera network but not seen. 
                env.cstate = [None, "00", 0]

            # state -> [curloc, preloc, time]]
        elif c1 and not c2: # IN CAM1
            env.cstate = ["01", None, None] # 여기도 duration으로 설명해야되나?
            
        elif not c1 and c2: # IN CAM2
            env.cstate = ["10", None, None]
            
        else: # OVERLAP
            print("no need to discuss yet. ")
            break
        print(env.cstate)
        env.writecumlativestates(count)

        # 2) action 
        env.chooseaction(env.scheme, reward)
        env.writeaction(count, env.action)

        # 3) reward
        reward = env.step(env.action, count) # 여기서 input으로 previous reward 계산 필요.
        #new_state, reward, done, _ = env.step(act) #--> 근데 여기에 return되는게 new_state가 나올수가 없는데 ㅠㅠ 

        env.writecumlativerewards(count, reward) 
        count += 1
        #print(count)
    #env.printcumlativeaction()
    #env.printcumlativereward()
    #env.printcumlativestates()

    print(env.sumrewards())

            
def setuplog():
    log = logging.getLogger('ohdeer_log')
    log.setLevel(logging.DEBUG)

if __name__ == '__main__':
    setuplog()
    play()