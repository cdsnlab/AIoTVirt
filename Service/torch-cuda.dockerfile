FROM mdegans/tegra-opencv

RUN mv /etc/apt/sources.list.d/nvidia-l4t-apt-source.list /etc/apt/
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates
RUN mv /etc/apt/nvidia-l4t-apt-source.list /etc/apt/sources.list.d

RUN apt update


RUN wget https://bootstrap.pypa.io/get-pip.py
# RUN wget https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN rm get-pip.py
# RUN pip install -U pip
RUN pip install setuptools Cython
#
# torch
#
ARG PYTORCH_URL=https://nvidia.box.com/shared/static/lufbgr3xu2uha40cs9ryq1zn4kxsnogl.whl
ARG PYTORCH_WHL=torch-1.2.0-cp36-cp36m-linux_aarch64.whl
# ARG PYTORCH_WHL=torch-1.5.0-cp36-cp36m-linux_aarch64.whl

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PYTORCH_URL} -O ${PYTORCH_WHL} && \
    pip install ${PYTORCH_WHL} --verbose && \
    rm ${PYTORCH_WHL}


RUN apt install python3-dev libopenblas-base libopenmpi-dev -y
#
# torchvision 0.4
#
ARG TORCHVISION_VERSION=v0.4.0
# ARG TORCHVISION_VERSION=v0.4.0
ARG PILLOW_VERSION=pillow<7
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"

RUN printenv && echo "torchvision version = $TORCHVISION_VERSION" && echo "pillow version = $PILLOW_VERSION" && echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"

RUN apt-get install -y --no-install-recommends \
		  git \
		  build-essential \
            libjpeg-dev \
		  zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*


RUN git clone -b ${TORCHVISION_VERSION} https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    python3 setup.py install && \
    cd ../ && \
    rm -rf torchvision && \
    pip install "${PILLOW_VERSION}"

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

# Service deps
RUN pip3 install nvidia-pyindex

# Grpc buiild 1.37.1 crashes gunicorn
COPY wheels/grpcio-1.37.0-cp36-cp36m-linux_aarch64.whl /wheels/.
RUN pip3 install /wheels/grpcio-1.37.0-cp36-cp36m-linux_aarch64.whl
# Tracking library
COPY wheels/motrackers-0.0.0-py3-none-any.whl /wheels/.
RUN pip3 install /wheels/motrackers-0.0.0-py3-none-any.whl

RUN pip3 install /wheels/grpcio-1.37.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install meinheld gunicorn flask tritonclient[all] msgpack requests msgpack-numpy aiohttp
RUN pip3 install numpy==1.19.4



COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY ./gunicorn_conf.py /gunicorn_conf.py

COPY ./app /app
COPY ./microservices /app/microservices
WORKDIR /app/

ENV PYTHONPATH=/app

EXPOSE 80

CMD [ "bash" ]