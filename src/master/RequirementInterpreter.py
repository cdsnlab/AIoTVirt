#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.Logger import Logger

def interpret(serviceInstance):
    serviceKnowledge = Knowledge()
    result = serviceKnowledge.interpret(serviceInstance)

    return result

class Knowledge(object):
    def __init__(self):
        self.logger = Logger()
        self.service = {
            "ObjectTracking": {
                "Performance": {
                    "DetectionSpeed": "ProcessingTime",
                    "DetectionAccuracy": "DetectionRate"
                },
                "Reliability": {
                    "VideoContinuity": "FPS"
                },
                "Security": {
                    "VideoComposition": "NumberOfComposedVideos"
                }
            },
            "ObjectCounting": {
                "Performance": {
                    "DetectionSpeed": "ProcessingTime",
                    "DetectionAccuracy": "DetectionRate"
                },
                "Reliability": {
                    "VideoContinuity": "FPS"
                },
                "Security": {
                    "VideoComposition": "NumberOfComposedVideos"
                }
            }
        }

        self.logger.debug("Get Service Knowledge")

    def interpret(self, serviceInstance):
        requirement = serviceInstance.getRequirement()
        serviceType = serviceInstance.getType()
        result = {

        }
        '''
       {
            "DetectionSpeed": {
                "metric": "ProcessingTime",              - for capability calculation
                "weight": 10,    // 0 ~ 10               - for utility function
                required: 70     // 0 ~ 100 (normalized) - for filtering
            }, ..
        }
        '''

        # Performance
        if requirement["Performance"] == "HIGH":
            if serviceType == "ObjectTracking":
                for key, val in self.service[serviceType]["Performance"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 10,
                        "required": 70
                    }
            if serviceType == "ObjectCounting":
                for key, val in self.service[serviceType]["Performance"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 10,
                        "required": 70
                    }
        elif requirement["Performance"] == "MID":
            if serviceType == "ObjectTracking":
                for key, val in self.service[serviceType]["Performance"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 6,
                        "required": 50
                    }
            if serviceType == "ObjectCounting":
                for key, val in self.service[serviceType]["Performance"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 6,
                        "required": 50
                    }
        else:
            if serviceType == "ObjectTracking":
                for key, val in self.service[serviceType]["Performance"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 4,
                        "required": 30
                    }
            if serviceType == "ObjectCounting":
                for key, val in self.service[serviceType]["Performance"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 4,
                        "required": 30
                    }

        # Reliability
        if requirement["Reliability"] == "HIGH":
            if serviceType == "ObjectTracking":
                for key, val in self.service[serviceType]["Reliability"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 10,
                        "required": 70
                    }
            if serviceType == "ObjectCounting":
                for key, val in self.service[serviceType]["Reliability"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 10,
                        "required": 70
                    }
        elif requirement["Reliability"] == "MID":
            if serviceType == "ObjectTracking":
                for key, val in self.service[serviceType]["Reliability"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 6,
                        "required": 50
                    }
            if serviceType == "ObjectCounting":
                for key, val in self.service[serviceType]["Reliability"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 6,
                        "required": 50
                    }
        else:
            if serviceType == "ObjectTracking":
                for key, val in self.service[serviceType]["Reliability"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 4,
                        "required": 30
                    }
            if serviceType == "ObjectCounting":
                for key, val in self.service[serviceType]["Reliability"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 4,
                        "required": 30
                    }

        # Security
        if requirement["Security"] == "HIGH":
            if serviceType == "ObjectTracking":
                for key, val in self.service[serviceType]["Security"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 10,
                        "required": 70
                    }
            if serviceType == "ObjectCounting":
                for key, val in self.service[serviceType]["Security"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 10,
                        "required": 70
                    }
        elif requirement["Security"] == "MID":
            if serviceType == "ObjectTracking":
                for key, val in self.service[serviceType]["Security"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 6,
                        "required": 50
                    }
            if serviceType == "ObjectCounting":
                for key, val in self.service[serviceType]["Security"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 6,
                        "required": 50
                    }
        else:
            if serviceType == "ObjectTracking":
                for key, val in self.service[serviceType]["Security"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 4,
                        "required": 30
                    }
            if serviceType == "ObjectCounting":
                for key, val in self.service[serviceType]["Security"].items():
                    result[key] = {
                        "metric": val,
                        "weight": 4,
                        "required": 30
                    }

        self.logger.debug("[InterpretedRequirement]")
        self.logger.debug(result)
        return result


