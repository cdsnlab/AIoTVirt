FROM python:3.6-slim

RUN python3 -m pip install flask pandas xlsxwriter requests
WORKDIR /app/code
COPY collector.py .

ENV FLASK_APP=collector.py
 
ENTRYPOINT [ "flask" ]
CMD ["run"]