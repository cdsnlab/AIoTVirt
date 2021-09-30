FROM tensorflow/tensorflow:latest-gpu

RUN apt update

RUN pip3 install meinheld gunicorn flask
RUN pip3 install nvidia-pyindex
RUN pip3 install tritonclient[all]
RUN pip3 install msgpack requests
# RUN pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install tensorflow-hub

RUN pip3 install opencv-python-headless
RUN pip3 install msgpack-numpy aiohttp line-profiler
# RUN pip3 install tritonclient==2.8.0 -U

WORKDIR /wheels
COPY ./wheels/motrackers-0.0.0-py3-none-any.whl .
RUN pip3 install motrackers-0.0.0-py3-none-any.whl

WORKDIR /

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY ./gunicorn_conf.py /gunicorn_conf.py

COPY ./app /app
COPY ./microservices /app/microservices
WORKDIR /app/

ENV PYTHONPATH=/app
ENV WEB_CONCURRENCY=1

EXPOSE 80

CMD [ "bash" ]