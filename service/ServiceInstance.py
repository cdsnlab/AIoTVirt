#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

class ServiceInstance(object):
    def __init__(self, payload):
        jsonObject = json.loads(payload)
        self.name = jsonObject['name']
        self.type = jsonObject['type']
        self.requestTime = jsonObject['requestTime']
        self.parameter = jsonObject['parameter']
        self.requirement = jsonObject['requirement']
        self.interpretedRequirement = {}
        self.selectedNodes = []

        print("--------------Service---------------")
        print("Name: "+self.name)
        print("Type: "+self.type)
        print("RequestTime: "+str(self.requestTime))
        print("Parameter: "+str(self.parameter))
        print("Requirement: "+str(self.requirement))
        print("-----------------------------------")

    def getRequirement(self):
        return self.requirement

    def getParameter(self):
        return self.parameter

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def getRequestTime(self):
        return self.requestTime

    def getInterpretedRequirement(self):
        return self.interpretedRequirement

    def setInterpretedRequirement(self, ir):
        self.interpretedRequirement = ir

    def getSelectedNodes(self):
        return self.selectedNodes

    def setSeledtedNodes(self, nodes):
        self.selectedNodes = nodes

    def makePayload(self):
        msg = {}
        msg["name"] = self.name
        msg["type"] = self.type
        msg["requestTime"] = self.requestTime
        msg["parameter"] = self.parameter
        msg["requirement"] = self.requirement
        msg["interpretedRequirement"] = self.interpretedRequirement
        msg["selectedNodes"] = self.selectedNodes

        jsonString = json.dumps(msg)

        return jsonString

