#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading

import paho.mqtt.client as mqtt
from pymongo import ReturnDocument

from service.CapabilityInstance import *


class ServiceCapabilityManager(threading.Thread):
    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, mosq, userdata, rc):
        print("Connected with result code " + str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("capability/"+self.serviceInstance.getName()+"/")

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        print("ServiceCapability is requested!")
        capabilityInstance = CapabilityInstance(str(msg.payload.decode('utf-8')))
        t = threading.Thread(target=self.updateCapability(capabilityInstance))
        t.start()

    def __init__(self, config, serviceInstance):
        threading.Thread.__init__(self)
        self.nodes = []
        self.serviceInstance = serviceInstance
        self.ip = config["MQTT"]["ip"]
        self.port = int(config["MQTT"]["port"])
        self.resourceDB = DBManager(config["Mongo"], "resource")

    def run(self):
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(self.ip, self.port, 60)
        client.loop_forever()

    def updateCapability(self, capabilityInstance):
        self.resourceDB.getCollection().find_one_and_update(
            {
                'node': capabilityInstance.getNode(),
                'name': capabilityInstance.getName()
            },
            {'$set': {
                'value': capabilityInstance.getValue()
            }}, return_document=ReturnDocument.AFTER
        )

    def availableNodes(self):
        self.nodes = [
            {
                'name': 'node01',
                'DetectionAccuracy': 100,
                'VideoComposition': 80,
                'VideoContinuity': 70,
                'DetectionSpeed': 90
            },
            {
                'name': 'node02',
                'DetectionAccuracy': 90,
                'VideoComposition': 40,
                'VideoContinuity': 70,
                'DetectionSpeed': 90
            }
        ]
        return self.nodes
