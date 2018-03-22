#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docker

from util.Logger import Logger


def startService(name):
    logger = Logger()
    logger.debug("Start Service!")
    client = docker.from_env()
    container = client.containers.run("face_detection:latest", detach=True)
    return container
