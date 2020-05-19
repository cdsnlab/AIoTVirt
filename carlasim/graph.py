import networkx as nx
import numpy as np
import queue
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path

def bfs_traversal(grid):
    G = nx.Graph()
    visited = np.zeros_like(grid)
    q = queue.Queue()

    height = len(grid)
    width = len(grid[0])

    q.put('0-0')
    # print("BFS")
    while q.empty() != True:
        row, col = get_index(q.get()) 

        if row < 0 or col < 0 or row >= height or col >= width or visited[row][col] == 1:
            continue

        visited[row][col] = 1
        for i in range(-1,2):
            nrow = row + i
            if nrow < 0 or nrow >= height:
                continue
            for j in range(-1, 2):
                ncol = col + j
                if ncol < 0 or ncol >= width or (ncol == 0 and nrow == 0):
                    continue
                if grid[nrow][ncol] == -1:
                    continue
                length = 1
                if ncol == 0 or nrow == 0:
                    length = math.sqrt(2)
                G.add_edge('{}-{}'.format(row, col), '{}-{}'.format(nrow, ncol), length=length)
                q.put('{}-{}'.format(nrow, ncol))
    return G


def get_index(element):
    arr = element.split('-')
    return int(arr[0]), int(arr[1])


def to_node(index):
    return "{}-{}".format(index[0], index[1])


def show_paths(matrix, paths):
    G = bfs_traversal(matrix)
    for i in range(13, len(matrix[0])):
        G.remove_node('{}-{}'.format(0, i))
    pos = dict( (n, get_index(n)) for n in G.nodes())
    labels = dict((n, n) for n in G.nodes())
    path = []
    for a in paths:
        path += a
    node_colors = ["blue" if n in path else "red" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos=pos, labels=labels, node_color=node_colors)
    nx.draw_networkx_edges(G, pos=pos)
    plt.show()