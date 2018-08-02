!/bin/bash
now=$(date "+%m-%d-%M-%S")
filename=$now.h264
raspivid -t 30000 -w 1920 -h 1080 -fps 20 -b 1200000 -p 0,0,720,680 -o $filename


