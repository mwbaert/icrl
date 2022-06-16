ARG BASE_CONTAINER=gitlab.ilabt.imec.be:4567/mwbaert/constraint-inference/icrl:latest

FROM $BASE_CONTAINER

USER root

#RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
#    libsm6 libxext6 libxrender-dev curl \
#    && rm -rf /var/lib/apt/lists/*

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN echo "**** Installing Python ****" && \
    add-apt-repository ppa:deadsnakes/ppa &&  \
    apt-get install -y build-essential python3.8 python3.8-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    rm -rf /var/lib/apt/lists/*

RUN sudo update-alternatives  --set python /usr/bin/python3.8
RUN sudo apt-get install libosmesa6-dev -y
RUN sudo apt-get install patchelf
RUN pip3.8 install --upgrade pip

COPY mujoco210-linux-x86_64.tar.gz /home/jovyan/mujoco210-linux-x86_64.tar.gz 
RUN tar -xvzf ~/mujoco210-linux-x86_64.tar.gz
RUN mkdir /home/jovyan/.mujoco/
RUN mv /home/jovyan/mujoco210 /home/jovyan/.mujoco/mujoco210
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jovyan/.mujoco/mujoco210/bin
RUN pip3.8 install gym wandb mpl-scatter-density glfw mujoco torch pandas tqdm imageio
RUN pip3.8 install mujoco-py


#RUN pip install stable-baselines3

#RUN git clone https://github.com/mwbaert/icrl.git
#RUN cd icrl
#RUN python -m pip install --upgrade pip
#RUN pip install -e ./custom_envs # To access custom environments through gym interface
#RUN wandb login fa44fb586bc0ae1f03502cb1b6268f3b916b163b
