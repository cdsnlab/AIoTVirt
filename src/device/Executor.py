#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docker

from util.Logger import Logger


def startService(name):
    logger = Logger()
    logger.debug("Start Service!")
    client = docker.from_env()
    client.containers.run("streaming:0.3", detach=True)
