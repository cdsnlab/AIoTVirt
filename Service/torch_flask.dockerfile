FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

RUN pip3 install pip --upgrade
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

ENV FLASK_APP=main.py
EXPOSE 80

CMD [ "flask run --port=80 --host=0.0.0.0" ]