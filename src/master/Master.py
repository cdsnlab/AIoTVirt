#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser

from master.ServiceManager import *

from master.RestServer import *


class Master(object):
    def __init__(self, configFile):
        logger = Logger()
        self.config = configparser.ConfigParser()
        self.config.read(configFile)

        logger.debug("Start Master!")
        serviceManager = ServiceManager(self.config)
        restService = RestServer(self.config['REST'], serviceManager)
