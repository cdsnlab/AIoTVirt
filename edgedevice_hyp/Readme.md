0. create directories

mkdir -p /home/$USER/workspace/redisdb


1. Run redis 

run by 

sudo docker run --name some-redis -d -p 6379:6379 -v /home/cdsn/workspace/redisdb:/data redis redis-server --appendonly yes

2. Run or build hyp

run by 

sudo docker run --rm --net=host -it -v /etc/apt/apt.conf:/etc/apt/apt.conf:ro --privileged -v /dev:/dev:shared -v /media/data2/NCS/:/media/data2/NCS/ ncs_hyp:latest /bin/bash 

or build it by

sudo docker build --tag spencerjang/ncs_hyp .
