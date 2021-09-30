FROM python:3.6-slim

RUN python3 -m pip install jetson-stats flask 
RUN python3 -m pip install numpy==1.19.4 
WORKDIR /app/code
COPY profiler.py .

ENV FLASK_APP=profiler.py

ENTRYPOINT [ "flask" ]
CMD ["run"]