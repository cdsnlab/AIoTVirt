#!/usr/bin/env python

# this program removes last n lines and saves it in place.
import cv2
import sys
import os
import time
import numpy as np
from mydarknet import Darknet
import json
import torch
import pickle as pkl
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
from torch.autograd import Variable
import pandas as pd
import argparse
import datetime
import dlib
import centroidtracker
import trackableobject
import threading
import tqdm
import requests

def remover(args):
    fp = args.fp
    count=0
    start_time = time.time()
    #for subdir, dirs, files in tqdm (list(os.walk(fp))):
    for subdir, dirs, files in os.walk(fp):       
        #print("reading csv in : {}".format(os.path.join(subdir)))
        if  os.path.isfile(subdir):
            pass
        else:
                        
            for i, file in enumerate(files): # camera files
                lst=[]
                if not '.csv' in file:
                    pass
                else:
                    print(file)
                    data = pd.read_csv(os.path.join(subdir,file)) 
                    #print(data.columns[data.isna().any()].tolist())
                    print(max(data.isnull().sum().tolist()))
                    data.drop(data.tail(max(data.isnull().sum().tolist())).index,inplace=True)
                    print("saving csv in : {}".format(os.path.join(subdir,file)))
                    data.to_csv(os.path.join(subdir,file))
                    #data.to_csv((file))

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description="welcome")
    argparser.add_argument(
        '--fp',
        metavar = 'f',
        #default = "/home/spencer/samplevideo/multi10zone_samplevideo/start_4_end_9_run_25/", # start_4_end_9_run_25
        default = "/home/spencer1/samplevideo/start1/",
        help='video folder location'
    )
    
    args = argparser.parse_args()
    # loop(args)
    remover(args)
