#!/bin/bash

arg=$1
echo "Argument: $1" > bbbbb.txt

for i in $(seq 1 100)
do
	echo "working on job $i"
	sleep 1
done
