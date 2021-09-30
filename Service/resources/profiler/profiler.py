from jtop import jtop
from flask import Flask
import socket
import threading

app = Flask(__name__)

PROFILE=False
thread = None
new_stats = None
numberofcores = 4
ourkeys = ['CPU1', 'CPU2', 'CPU3', 'CPU4', 'GPU', 'RAM', 'power cur']

@app.route('/start')
def start(): #start the jtop
    global PROFILE, thread
    PROFILE=True
    thread=threading.Thread(target=read_stats)
    thread.start()
    return "200"

@app.route('/stop')
def stop(): #stop the jtop
    global PROFILE, thread
    PROFILE=False
    thread.join()
    thread=None
    return "200"

def read_stats():
    global new_stats
    with jtop(1) as jetson:
        while jetson.ok() and PROFILE:
            stats = jetson.stats
            new_stats = {key: stats[key] for key in ourkeys}

@app.route('/return')
def serve(): #only returns values
    global new_stats
    
    met = logging(new_stats)
    met.update({"deviceid": socket.gethostname()})
    print(met)

    return met

def logging(stats):
    met = {}
    cpuvalues = 0
    for i in range(1,numberofcores+1):
        cpuvalues += stats['CPU{}'.format(i)]

    met['cpu'] = cpuvalues
    met['gpu'] = stats['GPU']
    met['ram'] = stats['RAM']
    met['pow'] = stats['power cur']

    return met
