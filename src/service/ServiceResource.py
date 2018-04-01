#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xmltodict
from flask import request
from flask_restful import Resource

from service.ServiceInstance import *
from util.Logger import Logger


class ServiceResource(Resource):
    def __init__(self, serviceManager):
        self.serviceManager = serviceManager

    def post(self, command):
        xml = request.data
        Logger().debug(xml)
        result = json.dumps(xmltodict.parse(xml)['service'])
        Logger().debug(result)

        if command == "start":
            self.serviceManager.receiveService(ServiceInstance(result))
        elif command == "stop":
            self.serviceManager.stopService(ServiceInstance(result))
        else:
            pass