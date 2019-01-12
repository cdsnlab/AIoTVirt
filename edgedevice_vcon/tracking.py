import os
import sys
import argparse
import time
import redis



def main():
    # do read the redis files.
    global r
    global window
    obj = ''
    #r = redis.Redis(host='localhost', port=6379, db=0)
    r = redis.Redis(host="localhost", port=6379, db=0)
    keys = r.keys('*')
    keylen = len(keys)
    for i in range (window):
        if r.hget(keylen-i, "numberofobjects") == 0: # 없는경우.
            pass
            # do smth.
        elif r.hget(keylen-i, "numberofobjects") !=0: # 여러개 있는 경우. 
            # keep track of all new objects.
            obj = r.hget(keylen-i, "a").decode("utf-8")
            #score, class, x, y로 구성 됨. 
            res = obj.strip('[]')
            data = res.split(',')
            print (data[3])


if __name__== '__main__':
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

