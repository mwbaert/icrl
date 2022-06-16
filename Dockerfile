ARG BASE_CONTAINER=gitlab.ilabt.imec.be:4567/mwbaert/constraint-inference/icrl:py38

FROM $BASE_CONTAINER

USER root

#EXPOSE 6006

#WORKDIR /project_scratch

#CMD ["/bin/bash"]

#RUN sudo apt-get upgrade -y
#RUN sudo apt-get update
RUN sudo update-alternatives  --set python /usr/bin/python3.8
RUN sudo apt-get install libosmesa6-dev -y
RUN sudo apt-get install patchelf
RUN pip install --upgrade pip

COPY mujoco210-linux-x86_64.tar.gz /home/jovyan/mujoco210-linux-x86_64.tar.gz 
#RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz ~/.mujoco/mujoco210
RUN tar -xvzf ~/mujoco210-linux-x86_64.tar.gz
RUN mkdir /home/jovyan/.mujoco/
RUN mv /home/jovyan/mujoco210 /home/jovyan/.mujoco/mujoco210
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jovyan/.mujoco/mujoco210/bin
RUN pip install gym wandb mpl-scatter-density glfw mujoco torch pandas tqdm imageio
RUN pip install mujoco-py


#RUN pip install stable-baselines3

#RUN git clone https://github.com/mwbaert/icrl.git
#RUN cd icrl
#RUN python -m pip install --upgrade pip
#RUN pip install -e ./custom_envs # To access custom environments through gym interface
#RUN wandb login fa44fb586bc0ae1f03502cb1b6268f3b916b163b
