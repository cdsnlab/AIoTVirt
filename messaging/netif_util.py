# -*- coding: utf-8 -*-
import subprocess


def get_netif_list():
    netif_list = []
    if_str = subprocess.check_output(['ifconfig']).decode('utf-8').lower()
    if_items = []
    idx_s = 0
    idx = if_str.find('\n\n')

    while idx != -1:
        if_items.append(if_str[idx_s:idx])
        idx_s = idx+2
        idx = if_str.find('\n\n', idx+1)

    for if_item in if_items:
        netif_info = {
                     'name': 'undefined',
                     'type': 'undefined',
                     'hwAddress': 'null',
                     'ipv4': 'null'
                 }

        if_item_split = if_item.split()
        # print(if_item_split[0])
        iface_name = if_item_split[0]
        if iface_name == 'lo':
            continue

        netif_info['name'] = iface_name
        if 'eth' in iface_name or 'br' in iface_name or iface_name.startswith('en'):
            netif_info['type'] = 'wired'
        elif 'wlan' in iface_name:
            netif_info['type'] = 'wifi'
        elif 'rmnet' in iface_name:
            netif_info['type'] = 'cellular'
        else:
            netif_info['type'] = 'undefined'

        idx_ipv4 = if_item.find('inet addr')
        if idx_ipv4 != -1:
            inet_addr_split = if_item[idx_ipv4+10:].split()
            # print(inet_addr_split[0])
            netif_info['ipv4'] = inet_addr_split[0]

        idx_hwaddr = if_item.find('hwaddr')
        if idx_hwaddr != -1:
            hwaddr_split = if_item[idx_hwaddr+6:].split()
            # print(hwaddr_split[0])
            netif_info['hwAddress'] = hwaddr_split[0]

        netif_list.append(netif_info)

    return netif_list


#
# Debug: test code
#
if __name__ == '__main__':
    res = get_netif_list()
    print(res)
