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
import matplotlib.pyplot as plt

# simulation parameters
DT = 0.1  # time tick [s]
SIM_TIME = 70.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range


# The main class for handling the robot drawing and geometry
class DrawRobotAndTrees(QWidget):
    def __init__(self, gui_world):
        super().__init__()
        self.title = "Robot and Orchard"

        # Window size
        self.top = 15
        self.left = 15
        self.width = 900
        self.height = 900
        self.scale = 10

        # State/action text
        self.sensor_text = "No sensor"
        self.action_text = "No action"
        self.move_text = "No move"
        self.loc_text = "No location"

        # The world state
        orchardWidth = 800
        orchardLength = 800
        orchard_startx = 0
        orchard_starty = 0
        self.orchard = Orchard.Orchard(0, 0, orchardWidth, orchardLength)

        """ Input 1 : Trees from initial orchard mapping"""
        trees_from_map = pd.read_csv('tree_df.csv')
        self.trees, self.tree_count, self.tree_matrix = self.orchard.add_trees(trees_from_map)

        # Robot Initialization
        """ Initial Robot state, set by user (Environmental modification) 
         We assume that the robot starts from a marked box on the  field at specified 
         initial heading for ease of localization. For now, we get the initial position 
         and heading from the ground truth data or GPS data         
         """
        self.ground_truth_from_gps = pd.read_csv('ply_gps.csv')
        self.gps_time = self.ground_truth_from_gps["field.header.stamp"]
        self.ground_x = self.ground_truth_from_gps["lat_utm"]
        self.ground_y = self.ground_truth_from_gps["lon_utm"]
        robot_startx = self.ground_x[0]
        robot_starty = self.ground_y[0]
        robot_start_heading = np.arctan2((self.ground_y[1] - self.ground_y[0]), (self.ground_x[1] - self.ground_x[0]))
        self.robot = Robot.Robot(robot_startx, robot_starty, robot_start_heading)

        # Particle Initialize
        self.particles_count = 1000
        self.particles = Particle.initialize_particles(self.robot.x_pf, self.robot.y_pf, self.robot.heading_pf,
                                                       self.particles_count)

        # Highlight detected tree
        self.tree_detected = None

        # Convergence plot
        self.convergence = []

    # What to draw
    def paintEvent(self, event):
        qp = QPainter()
        path = QPainterPath()
        qp.begin(self)
        self.draw_particles(qp, path)

        self.draw_orchard(qp)

        self.draw_robot(qp, path)
        qp.end()

    # Draw orchard
    def draw_orchard(self, qp):

        pen = QPen(Qt.black, 4, Qt.SolidLine)
        qp.setPen(pen)

        # Draw the Orchard boundaries
        qp.drawLine(0, 0, int(self.scale * self.orchard.orchard_width), 0)
        qp.drawLine(int(self.scale * self.orchard.orchard_width), 0, int(self.scale * self.orchard.orchard_width), int(self.scale * self.orchard.orchard_length))
        qp.drawLine(int(self.scale * self.orchard.orchard_width), int(self.scale * self.orchard.orchard_length), 0, int(self.scale * self.orchard.orchard_length))
        qp.drawLine(0, int(self.scale * self.orchard.orchard_length), 0, 0)


        # Draw the Trees
        pen = QPen(Qt.darkGreen, 14, Qt.SolidLine)
        qp.setPen(pen)
        print(self.tree_count)
        for j in range(self.tree_count - 1):
            qp.drawPoint(int(self.scale * self.trees[j].tree_x), int(self.scale * self.trees[j].tree_y))



        pen = QPen(Qt.darkBlue, 4, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawPoint(int(self.scale * self.tree_matrix.item(0)), int(self.scale * self.tree_matrix.item(1)))

        # pen = QPen(Qt.darkYellow, 4, Qt.SolidLine)
        # qp.setPen(pen)
        # if self.tree_detected:
        #     pen = QPen(Qt.darkRed, 4, Qt.SolidLine)
        #     qp.setPen(pen)
        #     for t in self.tree_detected:
        #         qp.drawPoint(t[0], t[1])

    def draw_robot(self, qp, path):

        pen = QPen(Qt.yellow, 2, Qt.SolidLine)
        qp.setPen(pen)

        """ Draw robot body as a point """
        robot_radius = 3
        pen = QPen(Qt.darkMagenta, 1, Qt.SolidLine)
        qp.setPen(pen)
        qp.setBrush(Qt.darkMagenta)
        center = QPoint(int(self.scale * self.robot.x), int(self.scale * self.robot.y))
        qp.drawEllipse(center, robot_radius, robot_radius)

        """ Draw robot header direction """
        pen = QPen(Qt.darkMagenta, 1, Qt.SolidLine)
        qp.setPen(pen)
        cos_val = np.cos(np.radians(self.robot.heading))
        sin_val = np.sin(np.radians(self.robot.heading))
        x_h = self.robot.x + (2 * robot_radius * cos_val)
        y_h = self.robot.y + (2 * robot_radius * sin_val)
        qp.drawLine(int(self.scale * self.robot.x), int(self.scale * self.robot.y), int(self.scale * x_h), int(self.scale * y_h))

        """ Draw Particle Filter Localized Robot position"""
        pen = QPen(Qt.darkCyan, 4, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawPoint(int(self.scale * self.robot.x_pf), int(self.scale * self.robot.y_pf))


        """ Draw Ground Truth Robot position"""
        pen = QPen(Qt.darkBlue, 4, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawPoint(int(self.scale * self.robot.x_gf), int(self.scale * self.robot.y_gf))

        pen = QPen(Qt.darkRed, 2, Qt.SolidLine)
        qp.setPen(pen)
        # text_loc = QPoint(self.orchard.orchard_startx + self.orchard.orchard_width + 10 , self.orchard.orchard_starty + 30)
        # text_loc = QPoint(self.orchard.orchard_startx,
        #                   self.orchard.orchard_starty + self.orchard.orchard_length + 30)
        # text_content = "Red - Ground Truth position of Robot" + str(self.robot.x) + str(self.robot.y)
        # qp.drawText(text_loc, text_content)
        #
        # pen = QPen(Qt.darkBlue, 2, Qt.SolidLine)
        # qp.setPen(pen)
        # text_loc = QPoint(self.orchard.orchard_startx + self.orchard.orchard_width + 10,
        #                   self.orchard.orchard_starty + 50)
        # qp.drawText(text_loc, "Blue - Dead reckoning estimate of Robot")
        #
        # pen = QPen(Qt.darkCyan, 2, Qt.SolidLine)
        # qp.setPen(pen)
        # text_loc = QPoint(self.orchard.orchard_startx + self.orchard.orchard_width + 10,
        #                   self.orchard.orchard_starty + 70)
        # qp.drawText(text_loc, "Cyan - Particle filter localisation")
        # text_loc = QPoint(20, self.orchard.orchard_starty + self.orchard.orchard_length + 20)
        # qp.drawText(text_loc, self.sensor_text)

        # For convergence plot
        # error = np.sqrt((self.robot.x - self.robot.x_pf) ** 2 + (self.robot.y - self.robot.y_pf) ** 2)
        # self.convergence.append(error)


    def draw_particles(self, qp, path):
        pen = QPen(Qt.lightGray, 3, Qt.SolidLine)
        qp.setPen(pen)
        for particle in self.particles:
            particle_in_window_x = int(particle.x - self.orchard.orchard_startx)
            particle_in_window_y = int(particle.y - self.orchard.orchard_starty)
            qp.drawPoint(self.scale * particle_in_window_x , self.scale * particle_in_window_y)

    def draw_plot(self):
        plt.figure(2)
        x_values = []
        for i in range(len(self.convergence)):
            x_values.append(i)
        plt.plot(x_values, self.convergence)
        plt.title("Convergence Plot for Particle Filter Localisation")
        plt.xlabel("Time step")
        plt.ylabel("Distance from ground truth(scale = y/10 ft)")
        plt.savefig("convergence_plot")


class StateEstimationGUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setWindowTitle('Orchard Localization')

        # Control buttons for the interface
        left_side_layout = self._init_left_layout_()
        middle_layout = self._init_middle_layout_()

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        # Three side-by-side panes
        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(middle_layout)

    # Set up the left set of sliders/buttons (state estimation)
    def _init_left_layout_(self):
        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)

        # Action/sensor buttons - update state estimation

        automate_motion = QPushButton('Start Simulation')
        automate_motion.clicked.connect(self.automate_motion)

        pause_motion = QPushButton('Pause Simulation')
        pause_motion.clicked.connect(self.pause_motion)

        restart_motion = QPushButton('Reset Simulation')
        restart_motion.clicked.connect(self.restart_motion)

        s_and_a = QGroupBox('State estimation: query state and do action')
        s_and_a_layout = QVBoxLayout()
        s_and_a_layout.addWidget(automate_motion)
        s_and_a_layout.addWidget(pause_motion)
        s_and_a_layout.addWidget(restart_motion)
        s_and_a.setLayout(s_and_a_layout)

        # Put all the pieces in one box
        left_side_layout = QVBoxLayout()

        # left_side_layout.addWidget(resets)
        left_side_layout.addWidget(s_and_a)
        left_side_layout.addStretch()

        return left_side_layout

    # Drawing screen and quit button
    def _init_middle_layout_(self):
        # The display for the robot drawing
        self.robot_scene = DrawRobotAndTrees(self)
        # self.particles_scene = DrawParticles(self, self.robot_scene.robot)

        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Put them together, quit button on the bottom
        mid_layout = QVBoxLayout()

        mid_layout.addWidget(self.robot_scene)
        # mid_layout.addWidget(self.particles_scene)
        mid_layout.addWidget(quit_button)

        return mid_layout

    # def robot_to_orchard_reference_frame(self, obj_loc):
    #     translation_x = self.robot_scene.robot.x
    #     translation_y = self.robot_scene.robot.y
    #     translation_matrix = np.matrix([[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]])
    #     trees_translation = np.matmul(translation_matrix, np.transpose(obj_loc))
    #     rotation_matrix = np.matrix([[np.cos(np.radians(self.robot_scene.robot.heading)),
    #                                   np.sin(np.radians(self.robot_scene.robot.heading)), 0],
    #                                  [-np.sin(np.radians(self.robot_scene.robot.heading)),
    #                                   np.cos(np.radians(self.robot_scene.robot.heading)), 0],
    #                                  [0, 0, 1]])
    #     trees_rotated = np.matmul(rotation_matrix, trees_translation)
    #     min = 1000
    #     trees_rotated_transpose = np.transpose(trees_rotated)
    #     tree_matrix_transpose = np.transpose(self.robot_scene.tree_matrix)
    #
    #     sensed_tree_matches = []
    #     for tree_est in trees_rotated_transpose:
    #         min = 1000
    #         for tree_ground in tree_matrix_transpose:
    #             d = np.sqrt(
    #                 (tree_ground.item(0) - tree_est.item(0)) ** 2 + (tree_ground.item(1) - tree_est.item(1)) ** 2)
    #             if d < min:
    #                 min = d
    #                 matched_tree = tree_ground
    #                 if d == 0:
    #                     break
    #
    #         sensed_tree_matches.append([matched_tree.item(0), matched_tree.item(1)])
    #     # return sensed_tree_matches
    #     return trees_rotated_transpose
    #
    # def get_imu_path(self):
    #     path = pd.read_csv('gps_path_heading.csv')
    #     # x,y = path['x'], path['y']
    #     return path

    def automate_motion(self):
        sim_i = 0
        tree_list = []
        hertz_track = 0

        """ Get the GPS Path """
        ground_truth = pd.read_csv('ply_gps.csv')
        ground_truth = ground_truth['field.header.stamp', 'lat_utm', 'lon_utm']
        print(ground_truth)
        input()

        """ Input 2 : Get IMU Data """
        dead_reckoning = pd.read_csv('imu_path_heading.csv')

        """ Input 3 : Get tree positions from robot ( from depth sensors) """
        waypoints = pd.read_csv('positions.csv')

        # while sim_i < len(path):
        #     sim_i += 1
        #     # if path.loc[sim_i, 'type'] == 'gyr':
        #     #     self.robot_scene.robot.set_heading(-path.loc[sim_i, 'heading_store'])
        #     # else:
        #     #     self.robot_scene.robot.move_dd(path.loc[sim_i, 'heading_store']*10)
        #     self.robot_scene.robot.set_heading(np.pi/4 - path.loc[sim_i-1, 'heading'])
        #     self.robot_scene.robot.move_dd(path.loc[sim_i-1, 'dd']*10)
        #     if sim_i % 10 == 0:
        #         self.repaint()
        #     # if hertz_track % 3 == 0:
        #     self.particle_filter_localisation()

    def pause_motion(self):
        print("pause")

    def restart_motion(self):
        print("restart")

    def draw(self, _):
        self.robot_scene.draw()

    def particle_filter_localisation(self):
        tree_locs, tree_distances = self.query_tree_sensor()
        if tree_locs:
            a, b, c = Particle.particle_filter(tree_locs, self.robot_scene.particles, tree_distances,
                                               self.robot_scene.orchard)
            self.robot_scene.particles = a
            self.robot_scene.robot.x_pf = b
            self.robot_scene.robot.y_pf = c
        self.repaint()

    def particle_filter_spread(self, delta_xy):
        distance_moved = np.hypot(delta_xy[0], delta_xy[1])
        error_range = 0.1 * distance_moved
        for p in self.robot_scene.particles:
            move_est = distance_moved + np.random.normal(-error_range, error_range)
            heading = self.robot_scene.robot.heading_est
            p.move_by(move_est, heading)

        self.robot_scene.robot.x_pf, self.robot_scene.robot.y_pf, confidence = Particle.compute_mean_point(
            self.robot_scene.particles)


if __name__ == '__main__':
    app = QApplication([])
    gui = StateEstimationGUI()
    gui.show()
    app.exec_()
