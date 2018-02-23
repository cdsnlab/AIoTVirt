#!/usr/bin/env python
# -*- coding: utf-8 -*-

from service.ServiceInstance import *
from master.ServiceCapabilityManager import *
import master.ResourceSelector as ResourceSelector
import master.RequirementInterpreter as RequirementInterpreter
import paho.mqtt.client as mqtt
from pymongo import MongoClient
import threading
from util.Logger import Logger

class ServiceManager(object):
    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, mosq, userdata, rc):
        print("Connected with result code " + str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("service/#")

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        print("Service is requested!")
        serviceInstance = ServiceInstance(str(msg.payload.decode('utf-8')))
        t = threading.Thread(target=self.publishService(serviceInstance))
        self.serviceList.append([t, serviceInstance])
        t.start()

    def __init__(self, ip, port):
        self.logger = Logger()
        self.serviceList = []

        self.ip = ip
        self.port = port

        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message

        client.connect(ip, port, 60)
        client.loop_forever()

    #
    # Service Management
    #

    '''
    Name: publishService
    parameter: ServiceInstance
    action: 
            let requirementInterpreter interpret service's requirements in terms of device's capabilities 
            -->
            let resourceSelector select suitable nodes which satisfy service's requirements
            -->
            let clusterManager make selected nodes start service
    '''
    def publishService(self, serviceInstance):
        print("PublishService starts!")
        requirement = serviceInstance.getRequirement

        # INTERPRET
        requirements = RequirementInterpreter.interpret()

        # SELECT
        serviceCapabilityManager = ServiceCapabilityManager(self.ip, self.port, serviceInstance)
        selectedNodes = ResourceSelector.selectNodes(serviceInstance, requirements)
        print("selected nodes: "+", ".join(selectedNodes))

        # START




