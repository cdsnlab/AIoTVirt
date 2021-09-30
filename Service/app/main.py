import sys
from time import time
import microservices as ms
from flask import Flask, request
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

app = Flask(__name__)

services = {}


@app.route("/link", methods=["POST"])
def link():
    requested_services = request.get_json()
    for service_name, args in requested_services.items():
        try:
            parent_kwargs = args.pop("common")
        except KeyError:
            parent_kwargs = {}
        service = ms.ms_from_str(service_name, args, parent_kwargs)
        services[service_name] = service

    for service_name, service in services.items():
        for link in service.outputs:
            service.link(services[link])

    return "200"


@app.route("/start", methods=["POST"])
def start():
    for service_name, service in services.items():
        if "Decoder" in service_name:
            service.start()
    return "200"


@app.route("/stop", methods=["DELETE"])
def stop():
    for service in services.values():
        service.cleanup()

    services.clear()

    return "200"


@app.route("/remote_src", methods=["POST"])
def remote_src():
    recv_time = time()
    d = request.get_data()
    send_time = float(request.headers.get("send-time")) # * Cast string back to float
    request_time = recv_time - send_time
    data = msgpack.unpackb(d)
    service_name = data.pop("service")
    inputs = data.pop("inputs")
    try:
        services[service_name].add_network_time(request_time)
        services[service_name].prepare(inputs)
    except KeyError:
        # print("Service not there")
        return "404"
    return "200"


@app.route("/metrics", methods=["GET"])
def metrics():
    metrics = {}
    for service_name, service in services.items():
        metrics[service_name] = service.get_metrics()
    
    return metrics


@app.route("/pipeline_metrics")
def pipeline_metrics():
    for _, service in services.items():
        if service.is_sink and service.metrics["pipeline"]["iters"] != 0:
            response = "Average pipeline time is {} \n Last 5 pipeline times are: {}".format(
                service.metrics["pipeline"]["total"] / service.metrics["pipeline"]["iters"],
                service.metrics["pipeline"]["list"][-5:]
            )
            
            print(response)
            
            return response

    return "No metrics"

@app.route("/queue_capacity")
def queue_capacity():
    queues = {}
    for service_name, service in services.items():
        queues[service_name] = service.get_queue_capacity()

    return queues

@app.route("/")
def hello():
    version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    message = "Hello World from Flask in a Docker container running Python {} with Meinheld and Gunicorn (default)".format(
        version
    )
    return message
