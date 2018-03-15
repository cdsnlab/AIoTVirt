#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
from flask import Flask
from flask_restful import Api
from service.ServiceResource import *
from util.Logger import Logger


class RestServer():
    def __init__(self, config, serviceManager):
        self.rest = config
        self.app = Flask(__name__)
        self.api = Api(self.app)
        self.api.add_resource(ServiceResource, "/service", resource_class_kwargs={'serviceManager': serviceManager})

        self.app.run(debug=True, host="0.0.0.0", port=int(self.rest["port"]))