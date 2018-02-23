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

