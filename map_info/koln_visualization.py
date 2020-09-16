import os

import numpy as np
import pandas as pd

# load dataset
#* porto dataset
#df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/porto_dataset/Porto_taxi_data_test_partial_trajectories.csv')

#* geolife dataset
# /home/spencer1/samplevideo/gps_datasets/geolife_dataset/Data/XXX/Trajectory/YYYY.plt
#df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/geolife_dataset/Data/')

#* koln dataset
#이거 longitude latitude 좌표가 아닌데?
#df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/koln.tr/koln.tr')

#* t-drive dataset
# XX 06 -> 14까지. 파일 명이 ID.
# '/home/spencer1/samplevideo/gps_datasets/t-drive_dataset/XX/YYYY.txt'

#location=[37.566345, 126.977893], # seoul
#location=[39.914991, 116.398286], # beijing
#location=[39.90745772086431, 116.35544946451571 ] # t-drive average


#df = pd.read_csv('/home/spencer1/samplevideo/gps_datasets/t-drive_dataset/')
#print(df.head())
allfiles=[]

for (path, dir, files) in os.walk("/home/spencer1/samplevideo/gps_datasets/t-drive_dataset/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            #print("%s/%s" % (path, filename))
            allfiles.append(path+"/"+filename)
#print(allfiles)

#* do the following to find the center of the map. 
# avgx, avgy, stdx, stdy= [], [], [], []
# avgx_iter, avgy_iter =[], []
# for file in allfiles: #* each ID
#     print(file)
#     if os.stat(file).st_size !=0: #* if not empty, read df.
#         df = pd.read_csv(file, header=None)
        
#         #first find the min max long, latitude
#         for index, row in df.iterrows():
#             if row[2] !=0.0 or row[3]!=0.0: 
#                 avgx_iter.append(row[2])
#                 avgy_iter.append(row[3])
#                 #updateminxmax(row[2], row[3])
#         if len(avgx_iter) >1 and len(avgy_iter) >1 :
#             avgx.append(sum(avgx_iter) / len(avgx_iter))
#             avgy.append(sum(avgy_iter) / len(avgy_iter))
#             stdx.append(statistics.stdev(avgx_iter))
#             stdy.append(statistics.stdev(avgy_iter))
#             #print(sum(avgx_iter) / len(avgx_iter), sum(avgy_iter) / len(avgy_iter), statistics.stdev(avgx_iter), statistics.stdev(avgy_iter))
#         avgx_iter, avgy_iter =[], []
# print(sum(avgx) / len(avgx), sum(avgy) / len(avgy), statistics.stdev(stdx), statistics.stdev(stdy))
tdrive_avg_lat = 39.90745772086431
tdrive_avg_lon = 116.35544946451571

allplots=[]
count=0
for file in allfiles: #* each ID
    if count == 100:
        break
    print(file)
    if os.stat(file).st_size !=0: #* if not empty, read df.
        df = pd.read_csv(file, header=None)
        
        #first find the min max long, latitude
        for index, row in df.iterrows(): #[2]->lon, [3]->lat
            if abs(row[2]-tdrive_avg_lon) < 2 and abs(row[3]-tdrive_avg_lat) <2:
                if row[2] !=0.0 or row[3]!=0.0: 
                    allplots.append([row[3], row[2]])
    count+=1

map_data = pd.DataFrame(
    allplots,
    columns=['lat', 'lon'],
    )

st.map(map_data, zoom=12)


#! for every time tick, it needs to print a point on the map... so there isn't a reason to draw it on the map... 
how should the datastructure be 
