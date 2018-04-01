#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docker
import configparser

from device.DeviceAbstractor import DeviceAbstractor
from util.Logger import Logger


class Device(object):
    def __init__(self, configFile):
        logger = Logger()
        self.config = configparser.ConfigParser()
        self.config.read(configFile)

        self.name = self.config['Information']['name']

        print(self.name)

        ip = self.config['MQTT']['ip']
        port = int(self.config['MQTT']['port'])
        duration = float(self.config['Profile']['duration'])

        self.location = float(self.config['Parameter']['location'])
        self.resolution = float(self.config['Parameter']['resolution'])
        logger.debug("Start Device!")


        deviceAbstractor = DeviceAbstractor(ip, port, duration, self.name)

        # deviceManager = DeviceManager(ip, port, self.name)







