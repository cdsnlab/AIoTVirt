# remove useless programs on raspi
sudo apt-get purge wolfram-engine -y
sudo apt-get purge libreoffice* -y 
sudo apt-get clean
sudo apt-get autoremove

# after network configuration is done...
sudo apt-get update && sudo apt-get upgrade

# install opencv library
sudo apt-get install build-essential cmake pkg-config -y

# image i/o packages to enable JPEG, PNG, TIFF
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev -y

# i/o packages for video streaming
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt-get install libxvidcore-dev libx264-dev -y

# GTK lib for GUI
sudo apt-get install libgtk2.0-dev libgtk-3-dev -y

# extra dependency
sudo apt-get install libatlas-base-dev gfortran -y

# python2.7 and python3 headers
sudo apt-get install python2.7-dev python3-dev -y

