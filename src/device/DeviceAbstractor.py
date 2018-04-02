#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import json
import paho.mqtt.publish as publish

from util.Logger import Logger

from device.Profiler import Profiler


class DeviceAbstractor(object):
    def __init__(self, ip, port, duration, name):
        self.logger = Logger()
        self.logger.debug("INTO DeviceAbstractor!")
        self.capabilityList = []

        self.ip = ip
        self.port = port
        self.duration = duration
        self.name = name

        self.profiler = Profiler(self.duration)

        self.doProfiling()

    def doProfiling(self):
        self.profiler.monitor_frame()

