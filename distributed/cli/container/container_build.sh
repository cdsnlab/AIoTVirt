#! /bin/bash

HOST_NAME=$(whoami)
HOST_UID=$(id -u $HOST_NAME)
HOST_GID=$(id -g $HOST_NAME)
CONTAINER_NAME=ligeti-cli
USER_NAME=ligeti


# #0: only build
# #1: only run
BUILD=$2
RUN=$3
DOCKERFILE=$1

if [[ "$DOCKERFILE" -eq "tf" ]]
    then
            DOCKERFILE="Dockerfile.tf"
            CONTAINER_NAME+=".tf"
    fi
if [[ "$BUILD" -eq 1 ]]
then
    docker build \
    --build-arg host_name=$USER_NAME \
    --build-arg host_gid=$HOST_UID \
    --build-arg host_uid=$HOST_GID \
    -f $DOCKERFILE \
    -t $CONTAINER_NAME .
fi
if [[ "$RUN" -eq 1 ]]
then
    docker run -t -d \
    --runtime nvidia \
    --gpus all \
    -u $HOST_UID \
    -v /home/$HOST_NAME/LIGETI:/home/$USER_NAME \
    -e HOME=/home/$USER_NAME/distributed/cli \
    -p 5000-5050:5000-5050 \
    --name=$CONTAINER_NAME \
    $CONTAINER_NAME
elif [[ "$RUN" -eq 2 ]]
then 
    docker run -it --rm \
    --runtime nvidia \
    --gpus all \
    -u $HOST_UID \
    -v /home/$HOST_NAME/LIGETI:/home/$USER_NAME \
    -e HOME=/home/$USER_NAME/distributed/cli \
    -p 5000-5050:5000-5050 \
    --name=$CONTAINER_NAME \
    $CONTAINER_NAME \
    /bin/bash
fi

# WORKDIR_ON_HOST=/home/$HOST_NAME/FedUSL/client/$HOST_NAME

# if [[ ! -d $WORKDIR_ON_HOST ]]
# then
#     mkdir $WORKDIR_ON_HOST
# else
#     echo "DIR exists."
# fi


# docker run -t -d \
# --runtime nvidia \
# --network host \
# --gpus all \
# -u $HOST_UID \
# -v /home/$HOST_NAME/FedUSL/client/$HOST_NAME:/home/$HOST_NAME \
# -e HOME=/home/$HOST_NAME \
# -p 5000-5050:5000-5050 \
# $CONTAINER_NAME
