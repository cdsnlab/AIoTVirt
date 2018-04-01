FROM sgtwilko/rpi-raspbian-opencv:stretch-3.3.1
RUN apt-get update && apt-get install -y libcap-dev
COPY requirements.txt /
RUN pip3 install -r requirements.txt
# I made some small modifications to imutils so I copy my version of it
# It would probably be more robust to do this changes in some other way
# but right now it is the only way that comes to my mind
# COPY overwrites files. Furthermore it doesn"t copy the directory itself
# but only its contents.
COPY imutils /usr/local/lib/python3.5/dist-packages/imutils/
COPY multiapp_async_test.py /
RUN mkdir cascades
COPY haarcascade_frontalface_default.xml haarcascade_eye.xml /cascades/
CMD ["python3", "multiapp_async_test.py"]
