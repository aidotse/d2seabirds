#FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.5.0-cudnn8-devel-ubuntu18.04
#FROM nvidia/cuda:11.5.0-cudnn8-devel-ubuntu20.04

# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        python3-opencv ca-certificates python3-dev \
        python3-pip git wget sudo ninja-build cmake
#RUN ln -sv /usr/bin/python3 /usr/bin/python

# ##RUN sudo apt-get update && sudo apt-get install -y vim
# ## create a non-root user
# ARG u_id
# ARG g_id
# ARG username

# RUN groupadd --gid ${g_id} ${username}
# RUN useradd --uid ${u_id} --gid ${g_id} --shell /bin/bash --create-home ${username}
# #RUN usermod -a -G video ${username}
# USER ${username}
# RUN chown -R ${u_id}:${g_id} /home/${username}
# RUN chmod -R  755 /home/${username}

# WORKDIR /home/${username}

#ARG USER_ID=1000
#RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER appuser
#WORKDIR /home/appuser
#ENV PATH="/home/appuser/.local/bin:${PATH}"
#RUN wget https://bootstrap.pypa.io/get-pip.py && \
#	python3 get-pip.py --user && \
#	rm get-pip.py

RUN python3 -m pip install --upgrade pip

#RUN pip install --user ipdb

#RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
##RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
#RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip3 install --user tensorboard
RUN pip3 install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#RUN pip3 install --user torch torchvision

RUN pip3 install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip3 install --user -e detectron2_repo
# Set dataset folder 
#ENV DETECTRON2_DATASETS="/home/juan.vallado/data/detectron2/"
#ENV DETECTRON2_DATASETS="/home/erik.svensson/local/data/detectron2/"

RUN pip3 uninstall jedi -y
RUN pip3 install --user jedi==0.17.2 
##RUN python -m pip install --user -e detectron2_repo
#RUN python -m pip install --user 'git+https://github.com/facebookresearch/detectron2.git'
RUN python3 -m pip install scikit-image

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
#WORKDIR /home/appuser/
#RUN chmod 777 /home/appuser/
#RUN git clone https://github.com/facebookresearch/detectron2
#RUN mv detectron2/projects/TensorMask .
#RUN rm -Rf detectron2
#RUN pip install -e TensorMask
#RUN chmod 777 TensorMask
#CMD sudo sh /home/juan.vallado/src/commands.sh
#CMD sudo sh /home/erik.svensson/git/ai_sweden/d2seabirds/src/commands.sh

WORKDIR /app