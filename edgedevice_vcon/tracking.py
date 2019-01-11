import os
import sys
import argparse
import time
import redis



def main():
    # do read the redis files.
    global r
    #r = redis.Redis(host='localhost', port=6379, db=0)
    r = redis.Redis(host="localhost", port=6379, db=0)
    keys = r.keys('*')
    keylen = len(keys)
    for i in range (5):
        print(r.hget(keylen-i, "a"))
        print(r.hgetall(keylen-i))


if __name__== '__main__':
    parser = argparse.ArgumentParser(
            description="tracking application container.")

    parser.add_argument( '-t', '--target', type=str,
            default='human',
            help="what to track.")

    ARGS = parser.parse_args()

    main()

