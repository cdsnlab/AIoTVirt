#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading

import paho.mqtt.client as mqtt

import device.Executor as Executor
from service.ServiceInstance import *
from util.Logger import Logger


class DeviceManager(object):
    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, mosq, userdata, rc):
        print("Connected with result code " + str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("device/#")

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        if( 'start' in msg.topic ):
            print("Service is allocated to device "+self.dev)
            serviceInstance = ServiceInstance(str(msg.payload.decode('utf-8')))
            t = threading.Thread(target=self.startService(serviceInstance))
            self.serviceList.append([t, serviceInstance])
            t.start()
        elif( 'stop' in msg.topic):
            pass

    def __init__(self, ip, port, dev):
        self.containers = []
        self.logger = Logger()
        self.serviceList = []

        self.ip = ip
        self.port = port
        self.dev = dev

        self.logger.debug("Device ["+self.dev+"] is connected!")

        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message

        client.connect(ip, port, 60)
        client.loop_forever()

    '''
    def startService(self, serviceInstance):
        serviceName = serviceInstance.getName()
        container = Executor.startService(serviceName)
        self.logger.debug(container.logs())
        self.containers.append({
            'serviceName': serviceInstance.getName(),
            'container': container
        })

    def stopService(self, serviceInstance):
        serviceName = serviceInstance.getName()
        self.containers.append({
            'serviceName': serviceInstance.getName(),
            'container': Executor.startService(serviceName)
        })
    '''

