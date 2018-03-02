#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paho.mqtt.client as mqtt
from util.Logger import Logger

class DeviceAbstractor(object):

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, mosq, userdata, rc):
        print("Connected with result code " + str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.

    def __init__(self, ip, port):
        self.logger = Logger()
        self.capabilityList = []

        self.ip = ip
        self.port = port

        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect

        self.client.connect(ip, port, 60)

    def changeCapability(self, cap, val):
        self.client.publish('capability/'+cap)