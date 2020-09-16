import os
import numpy as np
import pandas as pd

from pymongo import MongoClient
allfiles=[]

client = MongoClient('localhost', 27017)
db = client['tdrive']
mdb = db['plots'] 


for (path, dir, files) in os.walk("/home/spencer1/samplevideo/gps_datasets/t-drive_dataset/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            #print("%s/%s" % (path, filename))
            allfiles.append(path+"/"+filename)

tdrive_avg_lat = 39.90745772086431
tdrive_avg_lon = 116.35544946451571


count=0
for file in allfiles: #* each ID
    latplots=[]
    lonplots=[]

    allplots=[]

    print(file)
    if os.stat(file).st_size !=0: #* if not empty, read df.
        df = pd.read_csv(file, header=None)
        
        for index, row in df.iterrows(): #[2]->lon, [3]->lat
            if abs(row[2]-tdrive_avg_lon) < 2 and abs(row[3]-tdrive_avg_lat) <2:
                if row[2] !=0.0 or row[3]!=0.0: 
                    allplots.append([row[3], row[2]])
                    latplots.append(row[3])
                    lonplots.append(row[2])
        row = {"index": str(count), "lon": lonplots, "lat": latplots, "both": allplots}
        mdb.insert_one(row)
        print(row)
    count+=1