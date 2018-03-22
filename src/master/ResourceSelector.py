#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.Logger import Logger

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
    logger = Logger()
    print("SelectNodes start!")
    requirements = serviceInstance.getInterpretedRequirement()

    nodes = serviceCapabilityManager.availableNodes()

    result = []

    for node in nodes:
        result.append({'name': node['name'], 'utility': 0})
        for attr, req in requirements.items():
            resource = node[attr]
            if( resource < req["required"] ):
                logger.debug("Lack of resource ["+attr+"], required: ["+str(req["required"])+"], available: ["+str(resource)+"]")
                result.pop()
                break
            else:
                result[-1]['utility'] += resource * req['weight']

    logger.debug(result)


    return ['node01', 'node02']

