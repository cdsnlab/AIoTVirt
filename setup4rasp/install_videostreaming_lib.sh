# install imutils
sudo pip3 install imutils
sudo pip3 install opencv-python
sudo pip3 install python-prctl
# then... we don't need the previous installation???

# this is due to a change in imutils codes for camera thread naming. 
cd ../surveillance/misc
sudo cp pivideostream.py /usr/local/lib/python3.5/dist-packages/imutils/video
sudo cp videostream.py /usr/local/lib/python3.5/dist-packages/imutils/video
