import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from pymongo import MongoClient

#* each trip is classified as 'occupied, non-occupied' over time

allfiles=[]

client = MongoClient('localhost', 27017)
db = client['sf']
mdb = db['plots'] 


for (path, dir, files) in os.walk("/home/spencer1/samplevideo/gps_datasets/sf_dataset/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        #prefix = os.path.splitext(filename)[0]
        #print("new_" in os.path.splitext(filename)[0])
        if ext == '.txt' and "new_" in os.path.splitext(filename)[0]: # skip info files
            #print("%s/%s" % (path, filename))
            allfiles.append(path+"/"+filename)

sf_avg_lat = 37.765963
sf_avg_lon =  -122.433639

prevstate = 0
count=0
for file in tqdm(allfiles): #* each ID
    latplots=[]
    lonplots=[]
    allplots=[]

    print(file)
    if os.stat(file).st_size !=0: #* if not empty, read df.
        df = pd.read_csv(file, header=None, delimiter=" ")
        
        for index, row in df.iterrows(): 
            if abs(row[0]-sf_avg_lat) < 2 and abs(row[1]-sf_avg_lon) <2:
                if (int(row[2]) != prevstate): #* if this is the end of a sequence.
                    inputrow = {"index": str(count), "lon": lonplots, "lat": latplots, "both": allplots}
                    #print(inputrow)
                    #* save 
                    mdb.insert_one(inputrow)
                    count+=1
                    latplots=[]
                    lonplots=[]
                    allplots=[]
                    
                latplots.append(row[0])
                lonplots.append(row[1])
                allplots.append([row[0], row[1]])
                prevstate=int(row[2])
        latplots=[]
        lonplots=[]
        allplots=[]    

