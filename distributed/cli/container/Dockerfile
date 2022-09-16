FROM nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG host_uid
ARG host_gid
ARG host_name

RUN apt-get update ##[edited]
RUN apt install python3-pip -y
RUN pip3 install pip --upgrade
RUN apt install vim
RUN apt update
RUN apt install tzdata
RUN dpkg-reconfigure tzdata


# ENV PATH="/usr/local/cuda-10.2/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH}"
# RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

# RUN pip3 install pycuda six --verbose

RUN groupadd -g $host_gid $host_name
RUN useradd -u $host_uid -g $host_name $host_name
RUN usermod -aG video $host_name

USER $host_name

WORKDIR /home/$host_name/distributed/cli

CMD ["/bin/bash"]