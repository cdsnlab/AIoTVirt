FROM python:3.6-slim

RUN pip install tinydb streamlit requests numpy matplotlib graphviz kubernetes

WORKDIR /app

COPY *.py ./
COPY kubeconfig ./

CMD ["streamlit", "run", "metrics_dashboard.py"]