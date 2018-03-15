#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask_restful import Resource
from flask import request
from util.Logger import Logger
from service.ServiceInstance import *
import xmltodict, json

class ServiceResource(Resource):
    def __init__(self, serviceManager):
        self.serviceManager = serviceManager

    def post(self):
        xml = request.data
        Logger().debug(xml)
        result = json.dumps(xmltodict.parse(xml)['service'])
        Logger().debug(result)
        self.serviceManager.receiveService(ServiceInstance(result))