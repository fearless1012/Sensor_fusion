#!/usr/bin/env python3

# Get the windowing packages
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout, \
    QPushButton, QScrollArea
from PyQt5.QtCore import Qt, QSize, QPoint, QRect

from PyQt5.QtGui import QPainter, QPen, QPainterPath

import numpy as np
import Orchard
import Robot
import Particle
# import Robot_path_from_IMU as path
import matplotlib.pyplot as plt


def plotPaths(x, y, fig_num):
    # fig = plt.figure()
    # ax = plt.axes(xlim=(int(min(x)), int(max(x)) + 1), ylim=(int(min(y)), int(max(y)) + 1))
    # line, = ax.plot([], [], lw=2)
    #
    # def init():
    #     line.set_data([], [])
    #     return line,
    #
    # def animate(i):
    #     print(i)
    #     x_plot = []
    #     y_plot = []
    #     for j_ind in range(0, i * 200):
    #         if j_ind >= len(x):
    #             break;
    #         else:
    #             x_plot.append(x[j_ind])
    #             y_plot.append(y[j_ind])
    #     line.set_data(x_plot, y_plot)
    #     return line,
    #
    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                                frames=int(len(x) / 200), interval=2, blit=True)
    # # anim.save('coil.gif', writer='Pillow')
    #
    # plt.show()

    plt.figure(fig_num)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')


def particle_filter_localisation(self):
    tree_locs, tree_distances = self.query_tree_sensor()
    if tree_locs:
        a, b, c = Particle.particle_filter(tree_locs, self.robot_scene.particles, tree_distances,
                                           self.robot_scene.orchard)
        self.robot_scene.particles = a
        self.robot_scene.robot.x_pf = b
        self.robot_scene.robot.y_pf = c
    self.repaint()


if __name__ == '__main__':
    imu_path = pd.read_csv('imu_path_heading.csv')
    tree_positions = pd.read_csv('positions.csv')
    tree_positions['heading_store'] = tree_positions['data']
    tree_distances = tree_positions[['type', 'heading_store', 'field.header.stamp']]
    imu_path = imu_path[['type', 'heading_store', 'field.header.stamp']]

    imu_heading_treePositions = pd.concat([imu_path, tree_distances], axis=0, ignore_index=True).sort_values('field.header'
                                                                                                      '.stamp',
                                                                                                      ignore_index=True)

    imu_heading_treePositions.to_csv('imu_heading_treePositions.csv')
    # sim_i = 0
    # tree_list = []
    # hertz_track = 0
    # path = pd.read_csv('imu_heading_treeDistance.csv')
    # heading = 0
    #
    # x = []
    # y = []
    # x.append(0)
    # y.append(0)
    # temp_count = 0
    #
    # x_particle = []
    # y_particle = []
    # x_particle.append(0)
    # y_particle.append(0)
    #
    # particles = Particle.initialize_particles(x[0], y[0], heading)
    #
    # # The world state
    # orchardWidth = 800
    # orchardLength = 800
    # orchard_startx = 0
    # orchard_starty = 0
    # orchard = Orchard.Orchard(orchard_startx, orchard_starty, orchardWidth, orchardLength)
    # trees, tree_count, tree_matrix = orchard.add_trees()
    # trees_x = []
    # trees_y = []
    # tree_list = []
    #
    # for j in range(tree_count - 1):
    #     trees_x.append(trees[j].tree_x)
    #     trees_y.append(trees[j].tree_y)
    #     estimated_treexy = [trees[j].tree_x, trees[j].tree_y]
    #     tree_list.append(estimated_treexy)
    #
    # plt.scatter(trees_x, trees_y)
    #
    # while sim_i < len(path):
    #     print(sim_i)
    #     if path.loc[sim_i, 'type'] == 'gyr':
    #         heading = path.loc[sim_i, 'heading_store']
    #     elif path.loc[sim_i, 'type'] == 'acc':
    #         dd = path.loc[sim_i, 'heading_store']
    #
    #         x.append(x[len(x) - 1] + (dd * np.cos(heading)))
    #         y.append(y[len(y) - 1] + (dd * np.sin(heading)))
    #
    #         # x_particle.append(x_particle[len(x_particle) - 1] + (dd * np.cos(heading)))
    #         # y_particle.append(y_particle[len(y_particle) - 1] + (dd * np.sin(heading)))
    #
    #         error_range = 0.1 * dd
    #         for p in particles:
    #             move_est = dd + np.random.normal(-error_range, error_range)
    #             p.move_by(move_est, heading)
    #
    #         x_pf, y_pf, confidence = Particle.compute_mean_point(particles)
    #         x_particle.append(x_pf)
    #         y_particle.append(y_pf)
    #     else:
    #         tree_distances = []
    #         sensed_trees = []
    #         while path.loc[sim_i, 'type'] == 'dis':
    #             tree_distances.append(path.loc[sim_i, 'heading_store'])
    #             sim_i = sim_i + 1
    #         for d in tree_distances:
    #             min = 1000
    #             min_tree = tree_list[0]
    #             for tr in tree_list:
    #                 tr_dist_from_rob = np.sqrt((tr[0] - x_pf) ** 2 + (tr[1] - y_pf) ** 2)
    #                 if np.abs(d-tr_dist_from_rob) < min:
    #                     min = np.abs(d-tr_dist_from_rob)
    #                     min_tree = tr
    #             sensed_trees.append(min_tree)
    #         particles, x_pf, y_pf = Particle.particle_filter_sensor_fusion(sensed_trees, particles, tree_distances,
    #                                                                        orchard)
    #         x_particle.append(x_pf)
    #         y_particle.append(y_pf)
    #         sim_i = sim_i - 1
    #     sim_i = sim_i + 1
    # plotPaths(x, y, 2)
    # plotPaths(x_particle, y_particle, 3)
    # plt.show()
