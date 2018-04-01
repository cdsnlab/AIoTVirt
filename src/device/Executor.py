#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docker

from util.Logger import Logger


def startService(name):
    logger = Logger()
    logger.debug("Start Service!")
    client = docker.from_env()
    service = client.services.create("face_detection", name=name, networks=["swarm_net"],
                           mounts=["/home/pi/video/face_detection/container:/data:rw"], mode="replicated",
                           constraints=["node.labels.name==node03"])
    #container = client.containers.run("face_detection:latest", detach=True)
    return service
