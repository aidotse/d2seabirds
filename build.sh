#!/bin/bash

IMAGE_NAME="d2seabirds"

docker build -f Dockerfile -t $IMAGE_NAME .

# docker build \
# --build-arg u_id=$(id -u) \
# --build-arg g_id=$(id -g) \
# --build-arg username=$(id -gn $USER)  \
# -f Dockerfile -t $IMAGE_NAME .


#docker build \
#--build-arg u_id=$(id -u) \
#--build-arg g_id=$(id -g) \
#--build-arg username=$(id -gn $USER) \
#-f Dockerfile -t d2seabirds_erik .
