FROM python:3.7

RUN apt update

RUN pip3 install meinheld gunicorn flask
RUN pip3 install nvidia-pyindex
RUN pip3 install tritonclient[http]
RUN pip3 install msgpack requests

RUN pip3 install opencv-python-headless
RUN pip3 install msgpack-numpy
RUN pip3 install tritonclient==2.8.0 -U

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