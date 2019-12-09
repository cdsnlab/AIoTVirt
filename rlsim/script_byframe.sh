#!/bin/bash

for k in 0.3 0.5 0.7
do 
	for i in {0..10}
	do 
		python3 byframemulticamera.py --totit=20 --addarg=$i --like=$k
		sleep 2
	done
done
