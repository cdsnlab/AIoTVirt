from graphviz import Digraph
import json

# pip = {}
# with open("request.json", "r") as file:
#     pip = json.load(file)["requests"]

def get_dot_graph(pip, horizontal = False):

    cross_cluster = {}

    dot = Digraph(comment="Pipeline")
    dot.attr(bgcolor="transparent")
    if horizontal:
        dot.attr(rankdir="LR")

    for i, device in enumerate(pip):
        # * Want to have a subgraph for each device
        with dot.subgraph(name="cluster_device_" + str(i)) as c:
            c.attr(color='blue')
            c.attr(label='device_' + str(i), fontcolor="white")
            components = device["request"]
            not_local = set()
            for module, vals in components.items():
                # * Add Nodes IF they are not remote
                try:
                    is_local = vals["common"]["is_local"]
                    if is_local:
                        raise Exception()
                    not_local.add(module)
                except:
                    c.node(module, fontcolor="white")

            listed = list(components.items())
            for module, params in listed: #listed[:-2]:
                # * Add Edges between LOCAL nodes
                outs = []
                try:
                    outs = params["common"]["outputs"]
                except KeyError:
                    pass
                
                remote_outs = []
                for output in outs:
                    # * Check if output is local or not
                    if output not in not_local:
                        c.edge(module, output)
                    else:
                        remote_outs.append(output)

                if remote_outs:
                    cross_cluster[module] = remote_outs

            # * Should extract remote edges
            # last = listed[-2]
            # print(last)
            # cross_cluster[last[0]] = last[1]["common"]["outputs"]

    # * Add remote edges
    for start, ends in cross_cluster.items():
        for end in ends:
            dot.edge(start, end)

    return dot.source
