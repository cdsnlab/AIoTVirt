0) once stretch image is installed & network configuration is setup, do the following.
NOTE: network configuration is done manually along with port forwarding at the router. 

1) copy the two files. 
install_dependency.sh
install_opencv.sh
install_videostreaming_lib.sh

2) run the following codes to install opencv related libraries.
sudo chmod 777 install_dependency.sh
sudo chmod 777 install_opencv.sh
sudo chmod 777 install_videostreaming_lib.sh

sudo apt-get install tofrodos
sudo ln -s /usr/bin/fromdos /usr/bin/dos2unix 

dos2unix install_dep.sh
dos2unix install_opencv.sh
dos2unix install_videostreaming_lib.sh

./install_dependency.sh
./install_opencv.sh
./ install_videostreaming_lib.sh

3) run the video streaming.
cd surveillance/testing
python3 netstreamer_test.py