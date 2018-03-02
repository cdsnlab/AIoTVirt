#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
from device.DeviceManager import DeviceManager
from util.Logger import Logger

class Device(object):
    def __init__(self, ip, port, configFile):
        logger = Logger()
        config = configparser.ConfigParser()
        config.read(configFile)

        self.name = config['Information']['name']
        self.location = float(config['Parameter']['location'])
        self.resolution = float(config['Parameter']['resolution'])
        logger.debug("Start Device!")
        deviceManager = DeviceManager(ip, port, self.name)






