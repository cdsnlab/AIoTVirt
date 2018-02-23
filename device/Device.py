#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser

class Device(object):
    def __init__(self, configFile):
        config = configparser.ConfigParser()
        config.read(configFile)
        self.location = float(config['Parameter']['location'])
        self.resolution = float(config['Parameter']['resolution'])



