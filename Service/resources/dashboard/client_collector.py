import requests 
import argparse
import time

parser = argparse.ArgumentParser(description='set measure time.')
# Standard file to store the logs
# parser.add_argument('--file', dest="file", default="log.csv")
parser.add_argument('--dur', dest="dur", default=5)
args = parser.parse_args()

RESOURCE_COLLECTOR = None

# default_devices_list = {
#     "blue":   "http://192.168.1.181:9000/return",
#     "black":  "http://192.168.1.182:9000/return",
#     "green":  "http://192.168.1.183:9000/return",
#     "white":  "http://192.168.1.184:9000/return",
#     "orange": "http://192.168.1.185:9000/return",
#     "red":    "http://192.168.1.186:9000/return" 
# }

default_devices_list = {
    "jetson_boyan"  : "http://192.168.1.105:9000/return",
    "jetson_spencer": "http://192.168.1.128:9000/return"
}

def download(url, file_name):
    with open(file_name, "wb") as file:   # open in binary mode
        response = requests.get(url)
        print(response.headers)
        if response.headers['type']=="string":
            print("failed to download file")
            time.sleep(2)
            download(url, file_name)
        else:
            file.write(response.content)      # write to file
            print("file written")

def send_deviceupdate(data = default_devices_list):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(RESOURCE_COLLECTOR + '/update_devices', headers=headers, json=data)
    print("send_deviceupdate response: {}".format(response.content))

def send_start():
    response = requests.get(RESOURCE_COLLECTOR + '/start')
    print("send_start response: {}".format(response.content))

def send_stop():
    response = requests.get(RESOURCE_COLLECTOR + '/stop')
    print("send_stop response: {}".format(response.content))


def send_download(filename):
    url = RESOURCE_COLLECTOR + '/results/{}'.format(filename)
    download(url, "results/{}.xlsx".format(filename))


if __name__ == '__main__':
    #1. update device list.
    send_deviceupdate(default_devices_list)
    #2. send request xx second monitoring service. 
    send_start()
    #3. wait xx + alpha seconds
    time.sleep(int(args.dur)*2)
    print("sleeping : {}".format(int(args.dur)*2))
    send_stop()
    
    #4. return file.
    send_download("testss")

	
