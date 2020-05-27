ARG cuda_version=10.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

RUN apt-get update
RUN apt-get -qq -y install git curl build-essential subversion perl wget unzip vim

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.5

RUN python3.5 -V
RUN apt-get install -y python3-pip
RUN python3.5 -m pip install --upgrade pip
RUN ln -s /usr/bin/python3.5 /usr/bin/python
RUN pip install numpy grpcio tensorflow

COPY . /home/root/dist-dnn
WORKDIR /home/root/dist-dnn