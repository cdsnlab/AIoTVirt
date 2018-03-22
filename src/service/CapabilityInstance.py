#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

class CapabilityInstance(object):
    def __init__(self, payload):
        jsonObject = json.loads(payload)
        print(jsonObject)
        self.node = jsonObject['node']
        self.name = jsonObject['name']
        self.updateTime = jsonObject['updateTime']
        self.value = jsonObject['value']

        print("------------Capability--------------")
        print("Node: " + self.node)
        print("Name: " + self.name)
        print("UpdateTime: " + str(self.updateTime))
        print("Value: " + str(self.value))
        print("-----------------------------------")


    def getNode(self):
        return self.node

    def getName(self):
        return self.name

    def getUpdateTime(self):
        return self.updateTime

    def getValue(self):
        return self.value
