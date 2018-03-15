#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser

from device import DeviceManager
from util.Logger import Logger


class Device(object):
    def __init__(self, configFile):
        logger = Logger()
        self.config = configparser.ConfigParser()
        self.config.read(configFile)

        self.name = self.config['Information']['name']

        ip = self.config['MQTT']['ip']
        port = int(self.config['MQTT']['port'])

        self.location = float(self.config['Parameter']['location'])
        self.resolution = float(self.config['Parameter']['resolution'])
        logger.debug("Start Device!")
        deviceManager = DeviceManager(ip, port, self.name)






