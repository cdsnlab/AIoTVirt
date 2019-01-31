#!/usr/bin/env python
# -*- coding: utf-8 -*-
from node_info import NodeInfo


class NodeTable(object):
    def __init__(self):
        self.table = {}

    def add_entry(self, device_name, ip, port, location, capability):
        entry = NodeInfo(device_name, ip, port, location, capability)
        self.table[str(device_name)] = entry

    def remove_entry(self, device_name):
        self.table.pop(device_name, None)

    def get_entry(self, device_name):
        if device_name in self.table:
            return self.table[device_name]
        else:
            return None

    def get_list_str(self):
        t = []
        for key in self.table.keys():
            t.append(self.table[key])
        return str(t)


#
# Test code for NodeTable
#
if __name__ == '__main__':
    node_table = NodeTable()

    node_table.add_entry('Controller_N1_8F', '143.248.55.122', 8888, 'N1_8F', None)
    node_table.add_entry('Camera_N1_823_1', '143.248.55.122', 9999, '(10.1, 17.5)', None)
    node_table.add_entry('Camera_N1_823_2', '143.248.55.122', 9998, '(10.1, 18.8)', None)

    print(node_table.table)

    node_table.remove_entry('Camera_N1_823_1')
    print(node_table.table)

    node_info = node_table.get_entry('Controller_N1_8F')
    print(node_info.device_name, node_info.ip)
