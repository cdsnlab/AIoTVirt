Each folder contains...
docker_related -> docker related (dockerfile, build files) codes.
videofeed -> video streaming to edge server codes.



=============================INSTALLING========================================

1) once stretch image is installed & network configuration is setup, do the following.
NOTE: network configuration is done manually along with port forwarding at the router. 

note: read this blog for more information on installing opencv & dependency libraries. 
https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/

2) run the following codes to install opencv related libraries.
sudo chmod 777 install_dependency.sh
sudo chmod 777 install_opencv.sh
sudo chmod 777 install_videostreaming_lib.sh

sudo apt-get install tofrodos
sudo ln -s /usr/bin/fromdos /usr/bin/dos2unix 

dos2unix install_dependency.sh
dos2unix install_opencv.sh
dos2unix install_videostreaming_lib.sh

./install_dependency.sh
./install_opencv.sh
./install_videostreaming_lib.sh

3) run the video streaming under either... 
cd surveillance/testing
python3 netstreamer_test.py
or 
cd videofeed/xxxx.py

==========================REDIS UPDATE NEEDED==================================
wget http://download.redis.io/redis-stable.tar.gz 
tar xvzf redis-stable.tar.gz
cd redis-stable
make
sudo make install




