import time
import requests
from flask import Flask, request, send_file, make_response
import threading
import pandas as pd

app = Flask(__name__)

fields = ['count', 'deviceid', 'cpu', 'gpu', 'ram', 'pow']
devicepool = {}
threadlist = []
dataframes = {}
STATUS = False

@app.route("/update_devices", methods = ["POST", "GET"])
def update_devices():
    global devicepool
    #devicepool = {"blue": "http://192.168.1.181:9000/return","black": "http://192.168.1.182:9000/return","green": "http://192.168.1.183:9000/return","white": "http://192.168.1.184:9000/return","orange": "http://192.168.1.185:9000/return", "red": "http://192.168.1.186:9000/return" }
    
    devicepool = request.get_json()
    return str(devicepool)

@app.route("/start") 
def start():   
    global STATUS
    STATUS=True
    for device, deviceurl in devicepool.items():
        print("sending start command: {}".format(device))
        start_command(deviceurl)
        print("spwaning thread: {}".format(device))
        thread = threading.Thread(target=collect, args=(device, deviceurl,))
        threadlist.append(thread)
        thread.start()
        # thread.join(timeout=1)
    # time.sleep(2)
    # STATUS=False
    # time.sleep(2)
    # for thread in threadlist:
    #     thread.join()
    #     print("stopping ")
    return "200"

@app.route("/stop")
def stop():
    global STATUS
    STATUS=False

    for thread in threadlist:
        print("stopping...")
        thread.join()
    for device, deviceurl in devicepool.items():
        print("stopping jetson: {}".format(device))
        stop_command(deviceurl)

    return "200"

@app.route("/listdevices")
def listdevices():
    return str(devicepool)

@app.route("/results/<name>")
def getresults(name): #* get the latest results.

    filename = savetofile(name)
    time.sleep(1)
    if filename: #check if file exists
        print(filename)
        resp = make_response(send_file(filename, attachment_filename=filename, as_attachment=True))
        resp.headers["type"] = "file"
        return resp
    resp = make_response("Not ready")
    resp.headers["type"] = "string"
    return resp # return file

def savetofile(name):
    filename = 'resource_{}.xlsx'.format(name)
    # return send_file('log.csv', attachment_filename="log.csv", as_attachment=True)
    for thread in threadlist:
        if thread.isAlive():
            return False

    writer = pd.ExcelWriter(filename, engine='xlsxwriter', mode='w')
    for device, df in dataframes.items():
        df.to_excel(writer, sheet_name=device)
    writer.save()
    print("file saved as: {}".format(filename))
    threadlist.clear()
    return filename

def start_command(deviceurl):
    deviceurl=deviceurl.replace("return", "start")
    print(deviceurl)
    result=requests.get(deviceurl).json()
    print(result)

def stop_command(deviceurl):
    deviceurl=deviceurl.replace("return", "stop")
    print(deviceurl)
    result=requests.get(deviceurl).json()
    print(result)

def collect(device, deviceurl):
    
    count = 0
    rowlist = []
    while STATUS:
        time.sleep(1)
        data = requests.get(deviceurl).json()
        data.update({"count": count})
        print(data)
        rowlist.append(data)
        # writer.writerow(data)
        count+=1
    df = pd.DataFrame(rowlist)

    dataframes[device] = df
    
