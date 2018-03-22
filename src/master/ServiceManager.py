#!/usr/bin/env python
# -*- coding: utf-8 -*-

import master.ClusterManager as ClusterManager
import master.RequirementInterpreter as RequirementInterpreter
import master.ResourceSelector as ResourceSelector

from master.ServiceCapabilityManager import *
from util.Logger import Logger


class ServiceManager(object):
    def __init__(self, config):
        threading.Thread.__init__(self)
        self.logger = Logger()
        self.serviceList = []
        self.config = config

    def receiveService(self, serviceInstance):
        t = threading.Thread(target=self.publishService(serviceInstance))
        self.serviceList.append([t, serviceInstance])
        t.start()

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
        self.logger.debug("PublishService starts!")

        # INTERPRET
        interpretedRequirement = RequirementInterpreter.interpret(serviceInstance)

        # SELECT
        serviceInstance.setInterpretedRequirement(interpretedRequirement)
        serviceCapabilityManager = ServiceCapabilityManager(self.config, serviceInstance)
        serviceCapabilityManager.start()
        selectedNodes = ResourceSelector.selectNodes(serviceInstance, serviceCapabilityManager)

        print("selected nodes: "+", ".join(selectedNodes))
        self.logger.debug("selected nodes: "+", ".join(selectedNodes))

        # START
        serviceInstance.setSeledtedNodes(selectedNodes)
        ClusterManager.startService(self.config, serviceInstance)




