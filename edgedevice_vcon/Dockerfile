# dockerfile for container.
# input and output data?
#   - input: read redis values
#   - ouput: ???.

FROM python:3.6

# install libs.
RUN apt-get update \ 
	&& apt-get upgrade -y \
	&& apt-get install -y \ 
	build-essential sudo udev usbutils wget \
	&& apt-get clean all
RUN sudo pip3 install psutil
RUN sudo pip3 install redis
RUN sudo pip3 install paho-mqtt

ADD . /vcontainer
WORKDIR /vcontainer


CMD ["python3", "tracking.py"]
