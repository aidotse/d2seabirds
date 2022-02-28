#!/bin/bash

set -e

IMAGE_NAME="d2seabirds_erik"

ROOT_DIR=$PWD
#DATA_DIR=/home/erik.svensson/git/aise/d2seabirds/data
DATA_DIR=/home/erik/git/aise/d2seabirds/data
CODE_DIR=$ROOT_DIR/src

#export UID=$(id -u)
#export GID=$(id -g)

username=$(whoami)

#grep "^${username}:" /etc/passwd > .passwd.$$
#grep "^${username}:" /etc/group > .group.$$

#nvidia-docker run --rm \
#-e CUDA_VISIBLE_DEVICES=0 \
#-e CUDA_DEVICE_ORDER=PCI_BUS_ID \
#docker run --ipc=host -it --rm --runtime nvidia \
#-e CUDA_VISIBLE_DEVICES=0 \
docker run -it --ipc=host --rm \
-v $DATA_DIR:/home/$username/data \
-v $CODE_DIR:/home/$username/src \
-v $ROOT_DIR/output:/home/$username/output \
$IMAGE_NAME \
python3 src/data_processing_seabirds.py
#bash

#-it d2seabirds_erik \

#rm .passwd.$$
#rm .group.$$
