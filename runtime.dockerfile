# Base image with `pytorch` and `cuda`.
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

# Set working directory.
WORKDIR /elegant

# Update system.
RUN apt-get update

# Update pip.
RUN pip install -U pip

# Install C/C++ compilers (required dy `dlib`).
RUN apt-get install -y make g++

# Install CMake (required by `dlib`).
RUN pip install cmake

# Install missing dependencies.
RUN pip install opencv-python dlib fvcore matplotlib

# Install some more runtime dependencies.
RUN apt-get install -y git libglib2.0-0 libgl1-mesa-glx \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Default command is to login via `bash`.
CMD ["/bin/bash"]

