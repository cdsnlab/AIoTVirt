import math
import numpy as np
import time
import graph
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
import matplotlib.pyplot as plt
from matplotlib.path import Path
from itertools import islice
import random

# MAX_X = abs(-10)
# MAX_Y = abs(-6)
# MIN_X = abs(-68)
# MIN_Y = abs(-127)

# corners = [(-10, -126),
#            (-68, -127),
#            (-68, -8.5),
#            (-32.5, -8.5),
#            (-14.5, -25)]

# c = corners
#     # ,(-13, -6.5)]  This is the top right corner
# buildings = []
# buildings.append([(-31, -100), (-40.5, -100), (-40.5, -112), (-31, -112)])
# buildings.append([(-40.5, -118), (-51.5, -118), (-51.5, -103), (-40.5,-103)])
# buildings.append([(-52, -100), (-62, -100), (-62, -112), (-52, -112)])

# buildings.append([(-52, -74), (-62, -74), (-62, -85.5), (-52, -85.5)])
# buildings.append([(-52, -93), (-40.5, -93), (-40.5, -77), (-51.5, -77)])
# buildings.append([(-31, -74), (-40.5, -74), (-40.5, -86), (-31, -86)])

# buildings.append([(-60, -62), (-30, -62), (-30, -52), (-21, -41), (-36, -26), (-35, -24), (-44, -16.0), (-49, -16), (-60, -27)])

class Map(object):
    def __init__(self, merge=4):
        self.MAX_X = abs(-10)
        self.MAX_Y = abs(-6)
        self.MIN_X = abs(-68)
        self.MIN_Y = abs(-127)

        self.corners = [(-10, -126),
                (-68, -127),
                (-68, -8.5),
                (-32.5, -8.5),
                (-14.5, -25)]

        # c = corners
        self.buildings = []
        self.buildings.append([(-31, -100), (-40.5, -100), (-40.5, -112), (-31, -112)])
        self.buildings.append([(-40.5, -118), (-51.5, -118), (-51.5, -103), (-40.5,-103)])
        self.buildings.append([(-52, -100), (-62, -100), (-62, -112), (-52, -112)])

        self.buildings.append([(-52, -74), (-62, -74), (-62, -85.5), (-52, -85.5)])
        self.buildings.append([(-52, -93), (-40.5, -93), (-40.5, -77), (-51.5, -77)])
        self.buildings.append([(-31, -74), (-40.5, -74), (-40.5, -86), (-31, -86)])

        self.buildings.append([(-60, -62), (-30, -62), (-30, -52), (-21, -41), (-36, -26), (-35, -24), (-44, -16.0), (-49, -16), (-60, -27)])
        self.merge = merge
        self.matrix = self.create_map(self.MIN_X - self.MAX_X, self.MIN_Y - self.MAX_Y)
        self.outline_map()
        self.outline_buildings()
        self.G = graph.bfs_traversal(self.matrix)
        self.remove_top_row_nodes()


    def remove_top_row_nodes(self):
        if self.merge == 4:
            for i in range(13, len(self.matrix[0])):
                self.G.remove_node('{}-{}'.format(0, i))
        elif self.merge == 1:
            for i in range(13, len(self.matrix[0])):
                self.G.remove_node('{}-{}'.format(0, i))
            for i in range(38, len(self.matrix[0])):
                self.G.remove_node('{}-{}'.format(1, i))
            for i in range(64, len(self.matrix[0])):
                self.G.remove_node('{}-{}'.format(2, i))
            for i in range(89, len(self.matrix[0])):
                self.G.remove_node('{}-{}'.format(3, i))


    def show_graph(self, paths=[]):
        pos = dict( (n, graph.get_index(n)) for n in self.G.nodes())
        labels = dict((n, n) for n in self.G.nodes())

        if len(paths) != 0:
            path = []
            for a in paths:
                path += a
            node_colors = ["blue" if n in path else "red" for n in self.G.nodes()]
            nx.draw_networkx_nodes(self.G, pos=pos, labels=labels, node_color=node_colors)
        else:
            nx.draw_networkx_nodes(self.G, pos=pos, labels=labels)
        nx.draw_networkx_edges(self.G, pos=pos)
        plt.show()


    def get_line(self, start, end):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end

        >>> points1 = get_line((0, 0), (3, 4))
        >>> points2 = get_line((3, 4), (0, 0))
        >>> assert(set(points1) == set(points2))
        >>> print points1
        [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
        >>> print points2
        [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points


    def translate_to_index(self, posX, posY):
        """
        Given CARLA coordinates, translate to map/matrix indices
        """
        indexX = abs(math.floor((math.ceil(posX / self.merge) + math.floor(self.MAX_X / self.merge))))
        indexY = abs(math.floor((math.ceil(posY / self.merge) + math.floor(self.MIN_Y / self.merge))))
        return indexX, indexY


    def translate_to_coord(self, index, rand=True):
        """
        Given indices, translate to CARLA coordinates
        """
        row = index[0]
        col = index[1]
        if rand:
            posX = (row + random.random()) * self.merge + self.MAX_X
            if posX > 68.7:
                posX = random.uniform(67.5, 68.5)
            posY = (col + random.random()) * self.merge - self.MIN_Y
            if posY > -6.7:
                posY = random.uniform(-6.6, -7.3)
        else:
            posX = (row + 0.5) * self.merge + self.MAX_X  
            posY = (col + 0.5) * self.merge - self.MIN_Y
        return -posX, posY


    def create_map(self, rows, columns):
        """
        Create an empty numpy array representing the map given rows, columns and how many points are merged into a cell
        """
        return np.zeros((int(rows / self.merge) + 2, int(columns / self.merge) + 2), dtype=int)


    def outline_map(self):
        """
        Create the map outline using the corners from initialisation
        """
        for i in range(len(self.corners)):
            x, y = self.corners[i]
            row, col = self.translate_to_index(x, y)
            if i < len(self.corners) - 1:
                x, y = self.corners[i+1]
                nextX, nextY = self.translate_to_index(x, y)
            else:
                x, y = self.corners[0]
                nextX, nextY = self.translate_to_index(x, y)
            line = self.get_line((row, col), (nextX, nextY))
            # for pX, pY in line:
            #     self.matrix[pX][pY] = 1
            # self.matrix[row][col] = 1


    def outline_buildings(self):
        """
        Outline and fill buildings as obstacles on map
        """
        for building in self.buildings:
            
            corners = [self.translate_to_index(x, y) for x,y in building]
            path = Path(corners)
            minX = min(corners, key = lambda i : i[0])[0]
            minY = min(corners, key = lambda i : i[1])[1]
            maxX = max(corners, key = lambda i : i[0])[0]
            maxY = max(corners, key = lambda i : i[1])[1]

            for i in range(minX, maxX + 1):
                for j in range(minY, maxY +1):
                    if path.contains_point((i,j)):
                        self.matrix[i][j] = -1


            for i in range(len(building)):
                x, y = building[i]
                row, col = self.translate_to_index(x, y)
                if i < len(building) - 1:
                    x, y = building[i + 1]
                    nextX, nextY = self.translate_to_index(x, y)
                else:
                    x, y = building[0]
                    nextX, nextY = self.translate_to_index(x, y)
                line = self.get_line((row, col), (nextX, nextY))
                for pX, pY in line:
                    self.matrix[pX][pY] = -1


    def get_paths_coord(self, start, end, k=1):
        """
        Get (k) shortest paths from start to end
        """
        paths = self.get_paths_index(start, end, k)
        return [[self.translate_to_coord(graph.get_index(node)) for node in path] for path in paths]


    def get_paths_index(self, start, end, k=1):
        G = graph.bfs_traversal(self.matrix)
        for i in range(13, len(self.matrix[0])):
            G.remove_node('{}-{}'.format(0, i))
        pos = dict( (n, graph.get_index(n)) for n in G.nodes())
        labels = dict((n, n) for n in G.nodes())
        # print(G)
        # paths = nx.shortest_simple_paths(G, graph.to_node(start), graph.to_node(end))


        # for path in paths:
        #     print(path)
        path = nx.shortest_path(G, graph.to_node(start), graph.to_node(end))
        # path = nx.astar_path(G, graph.to_node(start), graph.to_node(end), self.dist)
        print(path)
        p = nx.node_disjoint_paths(G, graph.to_node(start), graph.to_node(end), flow_func=shortest_augmenting_path)
        path = []
        c = 0
        paths = []
        for a in p:
            paths.append(a)
            path += a
            c += 1
            if c == 1:
                break

        node_colors = ["blue" if n in path else "red" for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos=pos, labels=labels, node_color=node_colors)
        nx.draw_networkx_edges(G, pos=pos)

        # print(self.compare_paths(paths[0], paths[1]))
        # print(self.compare_paths(paths[1], paths[2]))
        # print(self.compare_paths(paths[0], paths[3]))
        # print(self.compare_paths(paths[2], paths[3]))
        plt.show()
        
        # plt.savefig('graph.png', dpi=1200)
            # break
        # return list(islice(paths, k))
        return [path]


    def get_shortest_paths(self, start, end, k=1):
        G = graph.bfs_traversal(self.matrix)
        for i in range(13, len(self.matrix[0])):
            G.remove_node('{}-{}'.format(0, i))
        pos = dict( (n, graph.get_index(n)) for n in G.nodes())
        labels = dict((n, n) for n in G.nodes())

        shortest = nx.astar_path(G, graph.to_node(start), graph.to_node(end), heuristic=self.dist)
        path = nx.shortest_simple_paths(G, graph.to_node(start), graph.to_node(end))
        return [shortest] + list(islice(path, k))


    def get_disjoint_paths(self, start, end):
        
        pos = dict( (n, graph.get_index(n)) for n in self.G.nodes())
        labels = dict((n, n) for n in self.G.nodes())


        # print(start)
        paths = nx.node_disjoint_paths(self.G, graph.to_node(start), graph.to_node(end), flow_func=shortest_augmenting_path)

        # path = []
        # for a in paths:
        #     path += a
        # node_colors = ["blue" if n in path else "red" for n in G.nodes()]
        # nx.draw_networkx_nodes(G, pos=pos, labels=labels, node_color=node_colors)
        # nx.draw_networkx_edges(G, pos=pos)
        # plt.show()
        return paths

    def get_graph(self):
        G = graph.bfs_traversal(self.matrix)
        return G

    def dist(self, a,b):
        ax, ay = graph.get_index(a)
        bx, by = graph.get_index(b)
        return abs(ax-bx) + abs(ay - by)

    def compare_paths(self, a, b):
        dist = 0
        for i in range(len(a)):
            # point_a = a[i]
            # point_b = b[i]
            try:
                dist += self.dist(a[i], b[i])
            except IndexError:
                pass
        return dist

# start = time.time()
# m = Map()

# print(m.translate_to_coord((14,0)))
# print(m.translate_to_coord((12,2)))
# print(m.translate_to_coord((6,0)))
# print(m.translate_to_coord((10,1)))
# print(m.translate_to_coord((14,2)))
# print(m.translate_to_coord((11,1)))


# m.get_paths_coord((2,2), (15,25))
# # print(m.get_paths_coord((0,0),
# (14,15), 1))
# G = m.get_graph()
# pos = dict( (n, graph.get_index(n)) for n in G.nodes())
# labels = dict((n, n) for n in G.nodes())
# # shortestPath = list(islice(nx.shortest_simple_paths(G, graph.to_node((0,0)), graph.to_node((14,14))), 1))
# # s = []
# # print(shortestPath)
# # for p in shortestPath:
# #     s += p
# # node_colors = ["blue" if n in s else "red" for n in G.nodes()]
# nx.draw_networkx_nodes(G, pos=pos, labels=labels)#, node_color=node_colors)
# nx.draw_networkx_edges(G, pos=pos)
# # print(time.time() - start)
# plt.show()