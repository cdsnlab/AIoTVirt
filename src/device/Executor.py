#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docker

from util.Logger import Logger


def startService(name):
    logger = Logger()
    logger.debug("Start Service!")
    client = docker.from_env()
    container = client.containers.run("streaming:0.3", ports={'1234/tcp': 1234}, devices='/dev/vchiq', privileged=True, detach=True)
    return container
