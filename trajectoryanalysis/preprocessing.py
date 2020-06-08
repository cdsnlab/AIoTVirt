"""
This program takes npy and tries to learn trajectories
"""


import os, sys, argparse, csv, math, time, json
from  pathlib import Path, PosixPath
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
import requests


def slacknoti(contentstr):
    
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


# format everything beforhand. 
def preprocessing(dataset, sr:int, vl:int, opt:str): # docnpy file path, cam id, training portion, variable length
    trainX, trainY = [], []
    testX, testY = [], []
    data = []
    for series, label in dataset:
        s = []
        for x,y in series:
            s.append(np.array([x / 2560, y / 1440]))
        data.append(np.array([s, label]))

    dataset = np.array(data)
    # slice ratio logic here.
    if opt == "sw-o": # sliding window complete overlap
        for series, label in dataset:   
            trainportion = int(len(series)/100*sr)
    
            #print("datasetlen {}".format(len(dataset)))     
            for i, (x, y) in enumerate(series):
                if i+vl > trainportion:
                    break
                #print(i, series[i:i+vl])
                trainX.append(series[i:i+vl])
                trainY.append(label) 

    elif opt == "sw-no": # sliding window no-overlap
        for series, label in dataset:
            trainportion = int(len(series)/100*sr)
            
            for i, (x, y) in enumerate(series):
                if i%vl ==0:
                    if i+vl > trainportion:
                        break
                    #print(i, series[i:i+vl])

                    trainX.append(series[i:i+vl])
                    trainY.append(label) 
                else:
                    pass
    
    elif opt == "ed": # evenly distributed 
        for series, label in dataset: 
            trainportion = int(len(series)/100*sr)

            tmpx, tmpy= [], []
            cnt = 0
            btw = math.floor(trainportion / float(vl))
            if trainportion < vl:
                continue
            for x, y in series[::btw]:
                if cnt == vl:
                    break
                else:
                    tmpx.append((x,y))
                cnt +=1
            tmpy.append(label)
            # append to trainX, trainY
            trainX.append(tmpx)
            trainY.append(label)

    elif opt == "irw" and sr!=100: # inverse reducing window but testing at different sr.
        portions = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        trainportion = int(len(series)/100*sr)

        for series, label in dataset:
            for prt in portions:
                startloc = int(len(series)/100*prt)
                tmpx, tmpy= [], []
                cnt = 0
                btw = math.floor((trainportion-startloc) / float(vl))
                if (trainportion-startloc) < vl: # if there are less plots than vl
                    continue
                for x, y in series[startloc:trainportion:btw]:

                    if cnt == vl:
                        break
                    else:
                        tmpx.append(np.array([x,y]))
                    cnt+=1
                #tmpy.append(label)
                if len(tmpx) < vl:
                    continue
                trainX.append(np.array(tmpx))
                trainY.append(label)

    elif opt == "irw": # inverse reducing window 10 - 100, 20 - 100, etc # for training
        portions = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        rwset = [90, 80, 70]
        for series, label in dataset:
            for prt in portions:
                startloc = int(len(series)/100*prt)
                for rw in rwset:
                    endloc = int(len(series)/100*rw)
                    btw = math.floor((endloc-startloc) / float(vl))
                    tmpx, tmpy= [], []
                    cnt = 0
                    if (endloc-startloc) < vl: # if there are less plots than vl
                        continue
                    for x, y in series[startloc:endloc:btw]:
                        if cnt == vl:
                            break
                        else:
                            tmpx.append(np.array([x,y]))
                        cnt+=1
                    #tmpy.append(label)
                    trainX.append(np.array(tmpx))
                    trainY.append(label)

    

    # trainX = np.array(trainX).reshape(len(trainX), vl*2)
    # trainY = np.array(trainY).reshape(len(trainY), )
    #testX = testX.reshape(len(testX), vl*2)
    #testY = testY.reshape(len(testY), )
    # return np.array(trainX), np.array(testX), np.array(trainY), np.array(testY)
    return np.array(trainX), np.array(trainY)

def learning(id, vl, trainX, trainY, testX, testY, method):
    #print(trainX.shape, trainY.shape)
    
    if trainX.size==0:
        print ("camera {} nothing here ".format(id))
        pass
    else:
        if method == "knn": # Nearest Neighbors
            clf = KNeighborsClassifier(n_neighbors = 1, n_jobs=4)
            clf.fit(trainX, trainY)
            #dump(clf, 'knn.joblib')
            predict_label = clf.score(testX, testY)
            print ('camera {} test accuracy {:.2f}'.format(id, predict_label))
            return predict_label

        elif method == "svm": # Linear SVM
            clf = SVC(kernel='linear', gamma="auto", verbose=False)
            clf.fit(trainX, trainY)
            #dump(clf, 'svm.joblib')
            predict_label = clf.score(testX, testY)
            print ('camera {} test accuracy {:.2f}'.format(id, predict_label))
            return predict_label

        elif method == "rf": # random forest
            clf = RandomForestClassifier(criterion = 'entropy', n_estimators=2, n_jobs=2, random_state=1)
            clf.fit(trainX, trainY)
            #dump(clf, 'rf.joblib')
            predict_label = clf.score(testX, testY)
            print ('camera {} test accuracy {:.2f}'.format(id, predict_label))
            return predict_label

        elif method=="ada": # adaboost
            clf = AdaBoostClassifier()
            clf.fit(trainX, trainY)
            #dump(clf, 'ada.joblib')

            predict_label = clf.score(testX, testY)
            print ('camera {} test accuracy {:.2f}'.format(id, predict_label))
            return predict_label

        elif method =="gnb": # naive bayes
            clf = GaussianNB()
            clf.fit(trainX, trainY)
            dump(clf, 'gnb.joblib')

            predict_label = clf.score(testX, testY)
            print ('camera {} test accuracy {:.2f}'.format(id, predict_label))
            return predict_label

        # ? not going to use 
        elif method =="qda": # quadratic discriminant analysis
            clf = QuadraticDiscriminantAnalysis()
            clf.fit(trainX, trainY)
            predict_label = clf.score(testX, testY)
            print ('camera {} test accuracy {:.2f}'.format(id, predict_label))
            return predict_label

        elif method == "mlp": # Neural Net
            clf = MLPClassifier(alpha=1, max_iter=1000)
            clf.fit(trainX, trainY)
            predict_label = clf.score(testX, testY)
            print ('camera {} test accuracy {:.2f}'.format(id, predict_label))
            return predict_label

        elif method=="gp": # gaussianprocess
            clf = GaussianProcessClassifier(1.0 * RBF(1.0))
            clf.fit(trainX, trainY)
            predict_label = clf.score(testX, testY)
            print ('camera {} test accuracy {:.2f}'.format(id, predict_label))
            return predict_label
        else:
            pass

def evaluate(fp, cam, srset, vl, opt, meth):
    #dataset=fp+str(cam)+".npy"
    dataset = np.load(fp+str(cam)+".npy", allow_pickle=True)
    X, Y= preprocessing(dataset, 100, vl, opt) # filepath, cam id, portion, option

    trainX, _, trainY, _  = train_test_split(X, Y, test_size=0.2, shuffle=False) # we might want to play with the input dataset.
    for sr in srset:
        X, Y= preprocessing(dataset, sr, vl, opt) # filepath, cam id, portion, option
        _, testX, _, testY  = train_test_split(X, Y, test_size=0.2, shuffle=False) # we might want to play with the input dataset.
        startime=time.time()                

        acc=learning(cam, vl, trainX, trainY, testX, testY, meth)
        endtime= time.time()

        result= {
            "cam":cam,
            "vl":vl,
            "sr":sr,
            "meth":meth,
            "opt":opt,
            "acc":acc,
            "time":endtime-startime
        }
        mypath = PosixPath("json/fixed_full/cam_{}_results.json".format(cam))
        #mypath = PosixPath("cam_{}_results.json".format(cam))

        json_data = None
        if mypath.exists():
            #with open("cam_{}_results.json".format(cam), 'r') as json_file:
            with open("json/fixed_full/cam_{}_results.json".format(cam), 'r') as json_file:
                json_data = json.load(json_file)
            json_data.append(result)
        else:
            json_data = [result]
        #with open("cam_{}_results.json".format(cam), 'w+') as json_file:
        with open("json/fixed_full/cam_{}_results.json".format(cam), 'w+') as json_file:
            json.dump(json_data, json_file)


def main():
    argparser = argparse.ArgumentParser(
        description="welcome")
    
    argparser.add_argument(
        '--fp',
        metavar = 'f',
        #default = "/home/spencer/samplevideo/multi10zone_npy/out/",
        #default = "npy_1500/",
        default = "npy/fixed_full/",
        help='video folder location'
    )
    argparser.add_argument(
        '--v',
        metavar = 'v',
        default = "30",
        type=int,
        help='input variable length'
    )
    argparser.add_argument(
        '--nc',
        metavar = 'nc',
        default = "10",
        type=int,
        help='number of cameras '
    )
    argparser.add_argument(
        '--tp',
        metavar = 'tp',
        type=int,
        default = "80",
        help='training proportion (%)'
    )
    argparser.add_argument(
        '--opt',
        metavar = 'opt',
        type=str,
        default = "sw-o", # ed: even distribution, sw: sliding window, 
        #default = "sw-o", # ed: even distribution, sw: sliding window, 
        help='preprocessing rule '
    )
    argparser.add_argument(
        '--meth',
        metavar = 'meth',
        type=str,
        default = "knn", # knn, svm, 
        help='learning '
    )
    argparser.add_argument(
        '--cam',
        metavar = 'cam',
        type=int,
        default = 0, # knn, svm, 
        help='siodjfoidsjifosjd '
    )

    args = argparser.parse_args()
    vl = args.v # input variable length
    fp = args.fp # csv file path
    nc = args.nc # csv file path
    opt = args.opt
    meth = args.meth
    cam = args.cam

    ##### qda fails requires some y to be more than 1
    ##### gp takes more than 1 hour for a small set of data
        
    methset = ["knn", "rf", "svm", "ada", "gnb"]
    #methset = ["knn", "rf", "svm", "gnb"]
    #methset = ["knn", "rf", "svm", "mlp", "gnb", "ada"] # mlp,gnb, gp, ada, rf, svm, knn

    vlset = [10]
    #vlset = [10, 20, 30]
    
    optset = ["sw-no"]
    #optset = ["ed", "sw-no", "rw"]
    
    srset = [80]
    #srset = [80, 60, 40, 20]
    
    cnt=0
    totallen = len(methset)*len(vlset)*len(optset)
    for opt in optset:
        for vl in vlset:            
            for meth in methset:
                print("{} / {} going with meth:{}, vl: {}, opt:{} at {} ".format(cnt, totallen, meth, vl, opt, time.strftime("%H:%M:%S", time.localtime())))
                slacknoti("[Mewtwo] {} / {} going with meth:{}, vl: {}, opt:{} cam: {} at {} ".format(cnt, totallen, meth, vl, opt,cam, time.strftime("%H:%M:%S", time.localtime())))
                evaluate(fp, cam, srset, vl, opt, meth)
                cnt+=1
    slacknoti("[Mewtwo] cam: {} done at {}".format(cam, time.strftime("%H:%M:%S", time.localtime())))

    
if __name__ == "__main__":
    main()