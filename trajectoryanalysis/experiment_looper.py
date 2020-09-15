import scenario_prop
import requests, time, csv, json

def slacknoti(contentstr):
    
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

# transitionmodels = ["conv_lstm"]
# timemodels = ["resnet", "dt", "rf", "svm", "hcf"]
# vls = [15, 30]
# preprocessingmethods = ["last", "sw-o", "ed", "irw"]

transitionmodels = ["conv_lstm"] #! waiting for tests. Start on Wed.
#timemodels = ["rf", "dt", "svm"]
timemodels = ["ResNet"]
#timemodels = ["hcf"] #passed: ResNet, rf, dt, svm
vls = [30] #* passed: 15, 30
# vls=[30]
preprocessingmethods = ["last"] #* passed: last, sw-o, ed, irw

for tim in timemodels:
    for trm in transitionmodels:
        for vl in vls:
            for pp in preprocessingmethods:
                sheetname = str(trm)+"_"+str(tim)+"_"+str(vl)+"_"+str(pp)
                scenario_prop.scenario_prop(vl, trm, tim, pp, sheetname)
    slacknoti("[MEWTWO] done with time model: {}, at {}".format(tim, time.strftime("%H:%M:%S", time.localtime())))