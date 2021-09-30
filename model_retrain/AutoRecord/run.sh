#!/bin/bash



SET=$(seq 1 3)
for i in $SET
do
echo "capture_cam_"$i".py"
python3 "capture_cam_"$i".py" &
done
