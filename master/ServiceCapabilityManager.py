#!/usr/bin/env python
# -*- coding: utf-8 -*-

from service.CapabilityInstance import *
import paho.mqtt.client as mqtt
import threading


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

    def __init__(self, ip, port, serviceInstance):
        threading.Thread.__init__(self)
        self.serviceInstance = serviceInstance
        self.ip = ip
        self.port = port

    def run(self):
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(self.ip, self.port, 60)
        client.loop_forever()

    def updateCapability(self, capabilityInstance):
        pass
