FROM mdegans/tegra-opencv

RUN mv /etc/apt/sources.list.d/nvidia-l4t-apt-source.list /etc/apt/
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates
RUN mv /etc/apt/nvidia-l4t-apt-source.list /etc/apt/sources.list.d
RUN apt install -y python3-pip

RUN pip3 install pip --upgrade
RUN pip3 install nvidia-pyindex
RUN pip3 install meinheld gunicorn flask tritonclient[http] msgpack requests msgpack-numpy aiohttp
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