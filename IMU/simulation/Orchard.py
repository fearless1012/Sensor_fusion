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
        return "(%f, %f, w=%f)" % (self.tree_x, self.tree_y, self.tree_id)


class Orchard:
    def __init__(self, start_x, start_y, width, length):
        self.orchard_width = width  # 400
        self.orchard_length = length  # 400
        self.orchard_startx = start_x  # (self.width - self.orchardWidth) / 2
        self.orchard_starty = start_y  # 0
        self.orchard_grid = np.zeros((self.orchard_width, self.orchard_length))  # flag and id

    def add_trees(self):
        x_loc = 10
        y_loc = 10
        trees = []
        tree_map_from_woo = pd.read_csv('gen_map_coords - Original.csv')
        # tree_map_from_woo = pd.read_csv('gen_map_coords.csv')
        tree_x = tree_map_from_woo['tree_x']
        tree_y = tree_map_from_woo['tree_y']
        # plt.scatter(tree_x, tree_y)
        # plt.show()
        tree_x_diff = tree_x.diff() * 10
        tree_y_diff = tree_y.diff() * 10

        number_of_trees = len(tree_map_from_woo)

        tree_list_x = []
        tree_list_y = []
        tree_list_1 = []

        # print("Orchard start = ", self.orchard_startx)

        for i in range(1, len(tree_map_from_woo)):
            tree_list_x.append(x_loc)
            tree_list_y.append(y_loc)
            tree_list_1.append(1)

            x_loc += tree_x_diff[i]
            y_loc += tree_y_diff[i]

        # min_x = min(tree_list_x)
        # min_y = min(tree_list_y)

        # for i in range(len(tree_list_x)):
        #     if min_x < 0:
        #         tree_list_x[i] += np.abs(min_x)
        #     if min_y < 0:
        #         tree_list_y[i] += np.abs(min_y)
        #     tree_list_x[i] += self.orchard_startx
        #     tree_list_y[i] += self.orchard_starty

        # -------------------------------------------------------------------------------------------
        # i = 0
        # while y_loc < 790:
        #     trees.append(Tree(i, x_loc, y_loc))
        #     tree_spacing = np.random.normal(15.0919, 1.9685)
        #     x_loc += tree_spacing
        #     if x_loc > 790:
        #         x_loc = tree_spacing
        #         row_spacing = np.random.normal(59.0551, 1.64042)
        #         y_loc += row_spacing
        #     i = i + 1
        # number_of_trees = i
        # -----------------------------------------------------------------------------------------------------

        # plt.scatter(tree_list_x, tree_list_y, label="0")
        tree_list = tree_list_x, tree_list_y, tree_list_1
        tree_matrix = np.matrix(tree_list)

        # Doing a rotation on the trees
        tan_rot_angle = (tree_list_y[len(tree_y) - 8] - tree_list_y[len(tree_y) - 5]) / (
                    tree_list_x[len(tree_y) - 8] - tree_list_x[len(tree_y) - 5])
        rotation_angle = -np.degrees(np.arctan(tan_rot_angle))
        rotation_angle = 90
        print("rotation ", rotation_angle)
        rotation_matrix = np.matrix([[np.cos(np.radians(rotation_angle)),
                                      -np.sin(np.radians(rotation_angle)), 0],
                                     [np.sin(np.radians(rotation_angle)),
                                      np.cos(np.radians(rotation_angle)), 0],
                                     [0, 0, 1]])
        trees_rotated = np.matmul(rotation_matrix, tree_matrix)
        trees_rotated_list = trees_rotated.tolist()
        trees_rotated_x = trees_rotated_list[0]
        trees_rotated_y = trees_rotated_list[1]

        # plt.legend()
        # plt.show()

        min_x = min(trees_rotated_x)
        min_y = min(trees_rotated_y)
        # print("minX", min_x)
        for i in range(len(trees_rotated_x)):
            if min_x < 0:
                trees_rotated_x[i] += np.abs(min_x)
            if min_y < 0:
                trees_rotated_y[i] += np.abs(min_y)

            trees_rotated_x[i] = trees_rotated_x[i] + self.orchard_startx + 10
            trees_rotated_y[i] = trees_rotated_y[i] + self.orchard_starty + 10

        tree_rotated_list = trees_rotated_x, trees_rotated_y, tree_list_1
        tree_rotated_matrix = np.matrix(tree_rotated_list)

        # print(tree_rotated_matrix)

        for i in range(len(trees_rotated_x)):
            trees.append(Tree(i, trees_rotated_x[i], trees_rotated_y[i]))

        return trees, number_of_trees, tree_rotated_matrix
