FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

WORKDIR /elegant

RUN pip install -U pip && pip install cmake

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y git libglib2.0-0 libgl1-mesa-glx \ 
     && apt-get clean \ 
     && rm -rf /var/lib/apt/lists/* 

RUN pip install -U pip && pip install opencv-python matplotlib dlib fvcore
