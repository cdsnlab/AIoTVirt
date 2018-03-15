#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paho.mqtt.client as mqtt
from util.Logger import Logger

def startService(config, serviceInstance):
    # The callback for when the client receives a CONNACK response from the server.
    logger = Logger()
    client = mqtt.Client()
    client.connect(config["MQTT"]["ip"], int(config["MQTT"]["port"]), 60)
    msg = serviceInstance.makePayload()
    logger.debug(msg)
    client.publish("device/start", msg)