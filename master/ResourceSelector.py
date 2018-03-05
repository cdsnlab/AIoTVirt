#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Name: selectNodes
parameter: ServiceInstance
action:
        let requirementInterpreter interpret service's requirements in terms of device's capabilities
        -->
        let resourceAllocator select suitable nodes which satisfy service's requirements
        -->
        let clusterManager make selected nodes start service
'''
def selectNodes(serviceInstance, serviceCapabilityManager):
    print("SelectNodes start!")
    requirement = serviceInstance.getInterpretedRequirement()

    return ['node01', 'node02']

