FROM nvcr.io/nvidia/deepstream:5.1-21.02-samples
WORKDIR /opt/nvidia/deepstream/deepstream-5.1/sources/
RUN apt update
RUN apt install python3-pip python-setuptools python3-gi gstreamer-1.0 -y
RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git