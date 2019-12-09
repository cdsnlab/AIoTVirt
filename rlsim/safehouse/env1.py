import gym
from gym import spaces
import random
import pandas
import numpy as np

action_size = 16 # N과 p의 조건에 의하면 정해지겠지?

#df = pandas.DataFrame (data = np.array([["GETCAM1","00", 0], ["GETCAM1","01", 2], ["GETCAM1","10", -1], ["GETCAM1","11", 1], ["GETCAM2","00", 0], ["GETCAM2","01", -1], ["GETCAM2","10", 2], ["GETCAM2","11", 1], ["GETBOTH","00", -2], ["GETBOTH","01", 1], ["GETBOTH","10", 1], ["GETBOTH","11", 2], ["GETNONE","00", 2], ["GETNONE","01", -1], ["GETNONE","10", -1], ["GETNONE","11", -2]]), columns=["action", "status", "reward"])
#df = [["GETCAM1","00", 0], ["GETCAM1","01", 2], ["GETCAM1","10", -1], ["GETCAM1","11", 1], ["GETCAM2","00", 0], ["GETCAM2","01", -1], ["GETCAM2","10", 2], ["GETCAM2","11", 1], ["GETBOTH","00", -2], ["GETBOTH","01", 1], ["GETBOTH","10", 1], ["GETBOTH","11", 2], ["GETNONE","00", 2], ["GETNONE","01", -1], ["GETNONE","10", -1], ["GETNONE","11", -2]]
#df = [["GETCAM1","00", 0], ["GETCAM1","01", 2], ["GETCAM1","10", 0], ["GETCAM1","11", 0], ["GETCAM2","00", 0], ["GETCAM2","01", 0], ["GETCAM2","10", 2], ["GETCAM2","11", 0], ["GETBOTH","00", 0], ["GETBOTH","01", 0], ["GETBOTH","10", 0], ["GETBOTH","11", 2], ["GETNONE","00", 2], ["GETNONE","01", 0], ["GETNONE","10", 0], ["GETNONE","11", 0]]
#df = [["01","00", 0], ["01","01", 2], ["01","10", 0], ["01","11", 0], ["10","00", 0], ["10","01", 0], ["10","10", 2], ["10","11", 0], ["11","00", 0], ["11","01", 0], ["11","10", 0], ["11","11", 2], ["00","00", 2], ["00","01", 0], ["00","10", 0], ["00","11", 0]]
df = [["01","00", -1], ["01","01", 2], ["01","10", -2], ["01","11", 1], ["10","00", -1], ["10","01", -2], ["10","10", 2], ["10","11", 1], ["11","00", -2], ["11","01", -1], ["11","10", -1], ["11","11", 2], ["00","00", 2], ["00","01", -2], ["00","10", -2], ["00","11", -2]]
ee = {}


class chamEnv(gym.Env):
    def __init__ (self, alpha, gamma, epsilon, gpeeweight):
        self.action_space =0 
        self.framenumber = None
        #self.alpha = alpha
        self.state = None # consist of a tuple. {}        
        self.qkey={}
        self.qvaluecount={}
        self.set_start_framenumber('1')
        self.perceivedstatus= None
        self.cumlativestates = {}
        self.cumlativeactions = {}
        self.cumlativerewards = {}
        self.cumgp = 0
        self.cumee = 0
        self.maxcumee =0
        self.cumgpdict = {}
        self.cumeedict = {}
        self.cstate = None
        self.curaction = "c"
        self.statushistory = []
        self.scheme = "random"
        self.actopt = ["01", "10", "11", "00"]
        self.action = None
        self.wrong = {} 

        #learn related parameters
        self.alpha = alpha
        #self.alpha = 0.9 # high: learning rate: how much i will accept the new value
        self.gamma = gamma
        #self.gamma = 0.9 # high: imideate reward
        self.epsilon = epsilon
        #self.epsilon = 0.1 # 낮을수록 좋게 설계함. 
        self.gpeeweight = gpeeweight
        #self.gpeeweight = 0.8 # high: gp, low: ee
        
    
    def createstateactionkeys(self, p, e): # creates [state & action] sets & initialize it
        # format [ '00', '00', integer ], {GETCAM1, GETCAM2, GETBOTH, GETNONE}
        # create 00, 01, 10, 11
        acloc = []
        aploc = []
        timer = []
        tmpstate = []
        # tmpact = ["GETCAM1", "GETCAM2", "GETBOTH", "GETNONE"] # this should be changed to binary too...
        # create curlocation 
        get_bin = lambda x, n: format(x, 'b').zfill(n)
        for i in range (0, pow(2,p)):
            acloc.append(get_bin(i,p))
            aploc.append(get_bin(i,p))
        acloc.append(None)
        aploc.append(None)
        # create dummy timer
        timer.extend(range(0, 40)) # time in seconds 
        timer.append(None)
        # adding curloc, prevloc, timer, action
        for i in acloc:
            for j in aploc:
                for k in timer:
                    for l in self.actopt:
                        tmpstate.append(str(i)+","+str(j)+","+str(k)+","+str(l))
        
        for i in tmpstate:
            self.qkey[i] = 0
        #print (self.qkey)


    def reset (self, p, e):
        print("[INFO] init state, reward, available action sets before starting")
        
        self.createstateactionkeys(p, e)
        
    def constructstate(self, curloc, prevloc, timer):
        return str(curloc)+","+str(prevloc)+","+str(timer)

    def deconstructstatefromstring(self, state):
        result = [x.strip() for x in state.split(',')]
        return result[0], result[1], result[2]

    def deconstructstatefromlist(self, state): # curloc, preloc, timer
        return state[0], state[1], state[2]

    def getmaxvalueofanaction(self, curloc, prevloc, timer):
        # get max value of state-action set. 
        curmax = -10
        for i in self.actopt:
            line = str(curloc)+","+str(prevloc)+","+str(timer)+","+str(i)
            
            if self.qkey[line] > curmax:
                #print("better pick: ", self.qkey[line])
                #print("line: ", line)
                curmax = self.qkey[line]
        return curmax

    def step (self, action, count): # input previous reward.
        gprew = 0
        eerew = 0
        # goodput + ee reward 
        for i in range (0, 16):
            if df[i][0] == action:
                if df[i][1] == self.perceivedstatus:
                    gprew = df[i][2]
                    if gprew > 0:
                        self.cumgp += 1 
                        self.wrong[count] = 0
                        break
                else:
                    self.wrong[count] =1

        # both < cam1 < cam2 < None 순으로 ee 점수가 높게 나와야 됨. 
        if action == "00":
            eerew = 2
            self.cumee += 0
        elif action == "01" or action == "10":
            eerew = 1
            self.cumee +=1
        else:
            eerew = 0
            self.cumee +=2
        self.maxcumee += 2

        rew = (self.gpeeweight * gprew) + ((1- self.gpeeweight) * eerew)
        
        # for e-greedy approach
        # q[state, action] = q[state, action] + alpha * (reward + gamma * np.max(q[new_state, :]) — Q[state, action])
        if count-1 is -1:
            acurstate, aprevstate, atimer = self.deconstructstatefromlist(self.cumlativestates[0])
            atmpkeystring = self.constructstate(acurstate, aprevstate, atimer)
            #print(self.qkey[atmpkeystring+","+action])
            maxvalue = self.getmaxvalueofanaction(acurstate, aprevstate, atimer)
            # need to get cumlative one.
            self.qkey[atmpkeystring+","+action] = self.qkey[atmpkeystring+","+action] + self.alpha * (rew + self.gamma * maxvalue - self.qkey[atmpkeystring+","+action])
            #print("current qvalue: ", self.qkey[atmpkeystring+","+action])
            self.qvaluecount[0] = self.qkey[atmpkeystring+","+action]
            self.cumeedict[0] = self.cumee 
            self.cumgpdict[0] = self.cumgp / 1 * 100
        else: 
            acurstate, aprevstate, atimer = self.deconstructstatefromlist(self.cumlativestates[count-1])
            atmpkeystring = self.constructstate(acurstate, aprevstate, atimer)
            #print(self.qkey[atmpkeystring+","+action])
            tcurstate, tprevstate, ttimer = self.deconstructstatefromlist(self.cumlativestates[count])
            ttmpkeystring = self.constructstate(tcurstate, tprevstate, ttimer)
            maxvalue = self.getmaxvalueofanaction(tcurstate, tprevstate, ttimer)
            
            self.qkey[atmpkeystring+","+action] = self.qkey[atmpkeystring+","+action] + self.alpha * (rew + self.gamma * maxvalue - self.qkey[atmpkeystring+","+action])
            #print("current qvalue: ", self.qkey[atmpkeystring+","+action])
            self.qvaluecount[count] = self.qkey[atmpkeystring+","+action]
            self.cumeedict[count] = self.cumee 
            self.cumgpdict[count] = self.cumgp / count * 100

        return rew
        
    def writecumlativestates (self, count):
        self.cumlativestates[count]= self.cstate

    def writecumlativerewards (self, count, reward):
        self.cumlativerewards[count]=reward

    def sumrewards(self):
        sumof = 0
        for n in range(len(self.cumlativerewards)):
            sumof += self.cumlativerewards[n]
        return sumof

    def writeaction(self, count, act):
        self.cumlativeactions[count] = act
        

    def printcumlativestates(self):
        #print(self.cumlativestates)
        print(*['{}: {}'.format(k, v) for k, v in self.cumlativestates.items()], sep='\n')

    def printcumlativereward(self):
        total = 0
        for i in range (0,len(self.cumlativerewards)):
            total = total + self.cumlativerewards[i]
        self.totreward = total
        
        print(self.cumlativerewards)
        print ("this is total", total)

    def printcumlativeaction(self):
        #print(self.cumlativeactions)
        #f = open('cumulativeaction.txt', mode='wt', encoding = 'utf-8')
        #f.write()
        print(*['{}: {}'.format(k, v) for k, v in self.cumlativeactions.items()], sep='\n')

    def setscheme(self, scheme):
        self.scheme = scheme

    def rindex(self, mylist, myvalue):
        try:
            len(mylist) - mylist[::-1].index(myvalue) - 1
            return len(mylist) - mylist[::-1].index(myvalue) - 1
        except ValueError:
            return False
        return len(mylist) - mylist[::-1].index(myvalue) - 1

    def translatestates(self, cam1, cam2):
        if cam1 and cam2: 
            self.perceivedstatus = "11"
        elif cam1 and not cam2:
            self.perceivedstatus = "01"
        elif not cam1 and cam2:
            self.perceivedstatus = "10"
        else:
            self.perceivedstatus = "00"
        self.statushistory.append(self.perceivedstatus)

    def getstatushistory(self): # search for all prev states, if it has 1 on which spot. 
        i1 = None
        i2 = None
        i1 = self.rindex(self.statushistory, "01")
        i2 = self.rindex(self.statushistory, "10")
        #둘중 하나라도 가장 최근에 있으면 리턴 값.
        #print (i1, i2)
        if i1 ==0 and i2 ==0:
            return "00"
        elif i1 <= i2:
            #self.statushistory[i2]
            return "10"
        elif i1 >= i2:
            #self.statushistory[i1]
            return "01"
       

    def setaction (self, action):
        self.curaction = action

    def chooseaction(self, scheme, count):
        # random.
        if scheme == "random":
            #print("choose from 1) getcam1, 2) getcam2, 3) getboth, 4) get none")
            self.action = random.choice (self.actopt)

        elif scheme == "eg":
            #print("do greedy action based on previous reward")
            # 현재 state에서 고를수 있는 값 중에서 가장 큰 값을 p의 확률로 고르고 나머지를 1-p확률로.
            acurstate, aprevstate, atimer = self.deconstructstatefromlist(self.cumlativestates[count])
            #atmpkeystring = self.constructstate(acurstate, aprevstate, atimer)
            newact = self.getactionofmax(acurstate, aprevstate, atimer)
            self.action = newact
        elif scheme =="rn":
            #print("do heuristic")
            acurstate, aprevstate, atimer = self.deconstructstatefromlist(self.cumlativestates[count])
            newact = self.getactionofrandmax(acurstate, aprevstate, atimer)

            self.action = newact
        else:
            print("else")
            self.action = "FIXMI"

    def getactionofmax(self, curloc, prevloc, timer):
        # get max value of state-action set. 
        randvalue=random.uniform(0,1)
        curmax = -1
        optopt = None
        for i in self.actopt:
            line = str(curloc)+","+str(prevloc)+","+str(timer)+","+str(i)
            if self.qkey[line] > curmax:
                curmax = self.qkey[line]
                print(line, curmax)

                optopt = i

        if randvalue > self.epsilon:
            print(optopt)
            return optopt
        else:
            newlist = [x for x in self.actopt if x != optopt]
            return random.choice (newlist)        

    def getactionofrandmax(self, curloc, prevloc, timer):
        # get max value of state-action set. 
        curmax = -1000
        optopt = None
        for i in self.actopt:
            line = str(curloc)+","+str(prevloc)+","+str(timer)+","+str(i)
            randval = random.randint(0, 5)
            if self.qkey[line] +randval> curmax:
                curmax = self.qkey[line] + randval
                print(line, curmax)

                optopt = i

        return optopt


    def set_start_framenumber(self, number):
        self.framenumber = number

    def show_episode(self):
        print("printing episode number")

    def show_frame_cnt(self):
        print("printing current frame number")

    def update_sim_status(self):
        print("updates simulation status, decides when to quit prog.")
