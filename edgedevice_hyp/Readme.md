run by 

sudo docker run --rm --net=host -it -v /etc/apt/apt.conf:/etc/apt/apt.conf:ro --privileged -v /dev:/dev:shared -v /media/data2/NCS/:/media/data2/NCS/ ncs_hyp:latest /bin/bash 

or build it by
sudo docker build --tag spencerjang/ncs_hyp .

