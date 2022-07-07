import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Tree:
    def __init__(self, idn, x, y):
        self.tree_id = idn
        self.tree_x = x
        self.tree_y = y

    def __repr__(self):
        return "(%f, %f, id=%f)" % (self.tree_x, self.tree_y, self.tree_id)


class Orchard:
    def __init__(self, start_x, start_y, width, length):
        self.orchard_width = width  # 400
        self.orchard_length = length  # 400
        self.orchard_startx = start_x  # (self.width - self.orchardWidth) / 2
        self.orchard_starty = start_y  # 0
        # self.orchard_grid = []  # flag and id

    def add_trees(self, trees_from_map):
        trees = []


        tree_x = trees_from_map['x']
        tree_y = trees_from_map['y']

        number_of_trees = len(trees_from_map)

        tree_list = tree_x, tree_y
        tree_matrix = np.matrix(tree_list)

        for i in range(len(tree_x)):
            trees.append(Tree(i, tree_x[i], tree_y[i]))


        min_x = min(tree_x)
        max_x = max(tree_x)
        min_y = min(tree_y)
        max_y = max(tree_y)

        self.orchard_startx = min_x - 50
        self.orchard_starty = min_y - 50
        self.orchard_width = max_x - min_x + 100
        self.orchard_length = max_y - min_y + 100
        # self.orchard_grid = np.zeros((int(self.orchard_width), int(self.orchard_length)))
        # for i in range(len(tree_x)):
        #     grid_x = int(tree_x[i] - self.orchard_startx)
        #     grid_y = int(tree_y[i] - self.orchard_starty)
        #     self.orchard_grid[grid_x][grid_y] = 1

        return trees, number_of_trees, tree_matrix
