#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docker
from util.Logger import Logger


def startService(serviceInstance):
    # The callback for when the client receives a CONNACK response from the server.
    logger = Logger()
    service = serviceInstance.getName()
    nodes = serviceInstance.getSelectedNodes()
    client = docker.from_env()

    for node in nodes:
        n = client.nodes.list(filters={'name': node})[0]
        config = {'Availability': 'active',
                 'Name': node,
                 'Labels': {service: 'true'}
                }
        if node == 'node01':
            config['Role'] = 'manager'
        else:
            config['Role'] = 'worker'
        n.update(config)
        n.reload()

    client.services.create("service", name=service, networks=["swarm_net"],
                                     mounts=["/home/pi/video/tracking/container:/data:rw"], mode="global",
                                     constraints=["node.labels."+service+"==true"])

def stopService(serviceInstance):
    # The callback for when the client receives a CONNACK response from the server.
    logger = Logger()
    service = serviceInstance.getName()
    client = docker.from_env()
    client.services.get(service).remove()