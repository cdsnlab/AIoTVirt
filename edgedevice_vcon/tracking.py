#-*-coding: utf-8 -*-
import os
import sys
import argparse
import time
import redis
import time
# mqtt messaging 
import message


def main():
    # do read the redis files.
    global r
    global window
    obj = ''

    message.sub_topic(message.client, "/data")

    r = redis.Redis(host="localhost", port=6379, db=0)
    keys = r.keys('*')
    keylen = len(keys)
    for i in range (window):
        if (r.hget(keylen-i, "numberofobjects") == 0):
            pass
            # do smth.
        elif r.hget(keylen-i, "numberofobjects") !=0: 
            obj = r.hget(keylen-i, "a").decode("utf-8")
            #score, class, x, y 
            res = obj.strip('[]')
            # do smth

    # send message to other devices.
    message.pub_message(message.client,"/data", "now?")
#    message.pub(client, "now?")

    time.sleep(10)

if __name__== '__main__':
    global window
    parser = argparse.ArgumentParser(
            description="tracking application container.")

    parser.add_argument( '-t', '--target', type=str,
            default='human',
            help="what to track.")
    parser.add_argument( '-w', '--window', type=int,
            default='10',
            help="window size of the tracking object.")

    ARGS = parser.parse_args()
    window = ARGS.window

    main()

