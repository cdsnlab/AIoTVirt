#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json


class NodeInfo(object):
    def __init__(self):
        self.device_name = 'default_device'
        self.ip = '0.0.0.0'
        self.port = 0
        self.location = 'N1'
        self.capability = 'no'

    def __init__(self, device_name, ip, port, location, capability):
        self.device_name = device_name
        self.ip = ip
        self.port = port
        self.location = location
        self.capability = capability

    def to_json(self):
        d = {'device_name': self.device_name, 'ip': self.ip, 'port': self.port, 'location': self.location,
             'capability': self.capability}
        return json.dumps(d, default=lambda o: o.__dict__)

    def __repr__(self):
        # d = {'device_name': self.device_name, 'ip': self.ip, 'port': self.port, 'location': self.location, 'capability': self.capability}
        return self.to_json()
