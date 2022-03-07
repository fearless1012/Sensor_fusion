#!/usr/bin/env python3

# Get the windowing packages
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout, \
    QPushButton, QScrollArea
from PyQt5.QtCore import Qt, QSize, QPoint, QRect

from PyQt5.QtGui import QPainter, QPen

import numpy as np
import Orchard
import Robot
import Particle
import matplotlib.pyplot as plt


# The main class for handling the robot drawing and geometry
class DrawRobotAndTrees(QWidget):
    def __init__(self, gui_world):
        super().__init__()

        # In order to get to the slider values
        self.gui = gui_world

        self.title = "Robot and Orchard"

        # self.scale = 2

        # Window size
        self.top = 15
        self.left = 15
        self.width = 1400
        self.height = 900

        # State/action text
        self.sensor_text = "No sensor"
        self.action_text = "No action"
        self.move_text = "No move"
        self.loc_text = "No location"

        # The world state
        self.orchard = Orchard.initialize_orchard(self.width)
        self.trees, self.tree_count, self.tree_matrix = Orchard.initialize_trees(self.orchard)

        # For querying tree
        self.sensor_triangle = []
        self.pointer_length = 196
        # self.tree_sensor = TreeSensor()

        # For moving robot
        # For robot state estimation
        self.robot = Robot.Robot(self.orchard)
        self.robot_estimation = Robot.RobotEstimation(self.robot, self.orchard)

        # For Particles
        self.particles = Particle.initialize_particles(self.robot_estimation)

        # For keeping sampled error
        self.last_wall_sensor_noise = 0
        self.last_move_noise = 0

        # Height of prob
        self.draw_height = 0.5

        # Set geometry
        self.text = "None"
        self.init_ui()

        # Convergence
        self.convergence = []

        # Highlight detected tree
        self.tree_detected = None
        self.tree_detected_estimate = None

        # backtracking
        self.prev_particle_tree_dist = np.zeros(1000)
        self.prev_robot_tree_dist = 1000
        self.robot_sensed_trees = []



    def init_ui(self):
        self.text = "Not reaching"
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    # For making sure the window shows up the right size
    def minimumSizeHint(self):
        return QSize(self.width, self.height)

    # For making sure the window shows up the right size
    def sizeHint(self):
        return QSize(self.width, self.height)

    # What to draw
    def paintEvent(self, event):

        qp = QPainter()
        qp.begin(self)

        self.draw_particles(qp)
        self.draw_sensor(qp)
        self.draw_orchard(qp)
        self.draw_robot(qp)
        qp.end()

    def draw_orchard(self, qp):
        pen = QPen(Qt.black, 4, Qt.SolidLine)
        qp.setPen(pen)
        # Orchard boundaries
        qp.drawLine(self.orchard.orchard_startx, self.orchard.orchard_starty,
                    self.orchard.orchard_startx + self.orchard.orchard_width, self.orchard.orchard_starty)
        qp.drawLine(self.orchard.orchard_startx + self.orchard.orchard_width, self.orchard.orchard_starty,
                    self.orchard.orchard_startx + self.orchard.orchard_width,
                    self.orchard.orchard_starty + self.orchard.orchard_length)
        qp.drawLine(self.orchard.orchard_startx + self.orchard.orchard_width,
                    self.orchard.orchard_starty + self.orchard.orchard_length, self.orchard.orchard_startx,
                    self.orchard.orchard_starty + self.orchard.orchard_length)
        qp.drawLine(self.orchard.orchard_startx, self.orchard.orchard_starty + self.orchard.orchard_length,
                    self.orchard.orchard_startx, self.orchard.orchard_starty)

        # Trees

        pen = QPen(Qt.darkGreen, 4, Qt.SolidLine)
        qp.setPen(pen)
        for j in range(self.tree_count):
            qp.drawPoint(self.trees[j].tree_x + self.orchard.orchard_startx,
                         self.trees[j].tree_y + self.orchard.orchard_starty)

        pen = QPen(Qt.darkYellow, 4, Qt.SolidLine)
        qp.setPen(pen)

        for t in self.robot_sensed_trees:
            qp.drawPoint(t[0], t[1])

        if self.tree_detected:
            pen = QPen(Qt.darkMagenta, 4, Qt.SolidLine)
            qp.setPen(pen)
            qp.drawPoint(self.tree_detected[0], self.tree_detected[1])

        if self.tree_detected_estimate:
            pen = QPen(Qt.darkBlue, 5, Qt.SolidLine)
            qp.setPen(pen)
            qp.drawPoint(self.tree_detected_estimate[0], self.tree_detected_estimate[1])

        # pen = QPen(Qt.darkYellow, 4, Qt.SolidLine)
        # qp.setPen(pen)
        # detected_trees = []
        # for particle in self.particles:
        #     if particle.detected_tree:
        #         detected_trees.append(particle.detected_tree.tree_id)
        #         qp.drawPoint(particle.detected_tree.tree_x, particle.detected_tree.tree_y)
        # counter = 0
        # tree_id = None
        # for i in detected_trees:
        #     curr_freq = detected_trees.count(i)
        #     if curr_freq > counter:
        #         counter = curr_freq
        #         tree_id = i

        # pen = QPen(Qt.green, 4, Qt.SolidLine)
        # qp.setPen(pen)
        # for tree in self.trees:
        #     if tree.tree_id == tree_id:
        #         qp.drawPoint(tree.tree_x, tree.tree_y)

    def draw_robot(self, qp):
        # display_robot_estimation(self, mode=""):

        pen = QPen(Qt.darkBlue, 2, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawPoint(self.robot_estimation.robot_est_x, self.robot_estimation.robot_est_y)

        pen = QPen(Qt.darkMagenta, 3, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawPoint(self.robot.x, self.robot.robot_y)

        error = np.sqrt((self.robot.robot_x - self.robot_estimation.robot_est_x) ** 2 + (
                self.robot.robot_y - self.robot_estimation.robot_est_y) ** 2)
        self.convergence.append(error)

    def draw_sensor(self, qp):
        pen = QPen(Qt.yellow, 2, Qt.SolidLine)
        qp.setPen(pen)

        cos_val = np.cos(np.radians(self.robot.robot_angle))
        sin_val = np.sin(np.radians(self.robot.robot_angle))
        cos_val_up = np.cos(np.radians(self.robot.robot_angle - 85))
        sin_val_up = np.sin(np.radians(self.robot.robot_angle - 85))
        cos_val_down = np.cos(np.radians(self.robot.robot_angle + 85))
        sin_val_down = np.sin(np.radians(self.robot.robot_angle + 85))
        x_p = self.robot.robot_x + (self.pointer_length * cos_val)
        y_p = self.robot.robot_y + (self.pointer_length * sin_val)
        x_p_up = self.robot.robot_x + (self.pointer_length * cos_val_up)
        y_p_up = self.robot.robot_y + (self.pointer_length * sin_val_up)
        x_p_down = self.robot.robot_x + (self.pointer_length * cos_val_down)
        y_p_down = self.robot.robot_y + (self.pointer_length * sin_val_down)
        # qp.drawLine(self.robot.robot_x, self.robot.robot_y, x_p, y_p)
        qp.drawLine(self.robot.robot_x, self.robot.robot_y, x_p_up, y_p_up)
        qp.drawLine(self.robot.robot_x, self.robot.robot_y, x_p_down, y_p_down)
        arc_rect = QRect(self.robot.robot_x - self.pointer_length, self.robot.robot_y - self.pointer_length,
                         self.pointer_length * 2, self.pointer_length * 2)
        qp.drawArc(arc_rect, (-self.robot.robot_angle) * 16, +85 * 16)
        qp.drawArc(arc_rect, (-self.robot.robot_angle) * 16, -85 * 16)
        # self.sensor_triangle = [self.robot.robot_x, self.robot.robot_y, x_p_up,y_p_up,x_p_down, y_p_down]

        pen = QPen(Qt.darkYellow, 2, Qt.SolidLine)
        qp.setPen(pen)
        text_loc = QPoint(20, self.orchard.orchard_starty + self.orchard.orchard_length)
        qp.drawText(text_loc, "Tree co-ordinates (from sensor) :")
        text_loc = QPoint(20, self.orchard.orchard_starty + self.orchard.orchard_length + 20)
        qp.drawText(text_loc, self.sensor_text)

    def draw_particles(self, qp, path):
        pen = QPen(Qt.lightGray, 3, Qt.SolidLine)
        qp.setPen(pen)
        for particle in self.particles:
            qp.drawPoint(particle.x, particle.y)

    # Map from [0,1]x[0,1] to the width and height of the window
    def x_map(self, x):
        return int(x * self.width)

    # Map from [0,1]x[0,1] to the width and height of the window - need to flip y
    def y_map(self, y):
        return self.height - int(y * self.height) - 1

    def in_pixels(self, v):
        return int(v * self.height)

    def draw_plot(self):
        plt.figure(2)
        x_values = []
        for i in range(len(self.convergence)):
            x_values.append(i)
        plt.plot(x_values, self.convergence)
        plt.xlabel("Time step")
        plt.ylabel("distance from ground truth")
        plt.show()

    def check_robot_estimation(self):
        if self.robot_estimation.robot_est_y > self.orchard.orchard_starty + 40:
            self.robot_estimation.robot_est_y = self.orchard.orchard_starty + 40
        elif self.robot_estimation.robot_est_y < self.orchard.orchard_starty + 20:
            self.robot_estimation.robot_est_y = self.orchard.orchard_starty + 20
        if self.robot_estimation.robot_est_x < self.orchard.orchard_startx:
            self.robot_estimation.robot_est_y = self.orchard.orchard_startx


class StateEstimationGUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('Orchard Localization')

        # Control buttons for the interface
        left_side_layout = self._init_left_layout_()
        middle_layout = self._init_middle_layout_()
        # right_side_layout = self._init_right_layout_()

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        # Three side-by-side panes
        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(middle_layout)
        # top_level_layout.addLayout(right_side_layout)

        # self.set_probabilities()

        # So the sliders can update robot_scene
        # SliderIntDisplay.gui = self
        # SliderFloatDisplay.gui = self

    # Set up the left set of sliders/buttons (state estimation)
    def _init_left_layout_(self):
        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)
        # scroll.setWidget()
        # Two reset buttons one for probabilities, one for trees
        # reset_probabilities_button = QPushButton('Reset probabilities')
        # reset_probabilities_button.clicked.connect(self.reset_probabilities)

        # reset_random_trees_button = QPushButton('Random trees')
        # reset_random_trees_button.clicked.connect(self.random_trees)

        # resets = QGroupBox('Resets')
        # resets_layout = QVBoxLayout()
        # # resets_layout.addWidget(reset_probabilities_button)
        # # resets_layout.addWidget(reset_random_trees_button)
        # resets.setLayout(resets_layout)

        # Action/sensor buttons - update state estimation
        query_tree_sensor_button = QPushButton('Query tree sensor')
        query_tree_sensor_button.clicked.connect(self.query_tree_sensor)

        move_left_button = QPushButton('Move left')
        move_left_button.clicked.connect(self.move_left)

        move_right_button = QPushButton('Move right')
        move_right_button.clicked.connect(self.move_right)

        move_up_button = QPushButton('Move up')
        move_up_button.clicked.connect(self.move_up)

        move_down_button = QPushButton('Move down')
        move_down_button.clicked.connect(self.move_down)

        rotate_robot_clockwise = QPushButton('Rotate Clockwise')
        rotate_robot_clockwise.clicked.connect(self.rotate_clockwise)

        rotate_robot_anticlockwise = QPushButton('Rotate Anticlockwise')
        rotate_robot_anticlockwise.clicked.connect(self.rotate_anticlockwise)

        automate_motion = QPushButton('Automate motion')
        automate_motion.clicked.connect(self.automate_motion)

        s_and_a = QGroupBox('State estimation: query state and do action')
        s_and_a_layout = QVBoxLayout()
        s_and_a_layout.addWidget(query_tree_sensor_button)
        s_and_a_layout.addWidget(move_left_button)
        s_and_a_layout.addWidget(move_right_button)
        s_and_a_layout.addWidget(move_up_button)
        s_and_a_layout.addWidget(move_down_button)
        s_and_a_layout.addWidget(rotate_robot_clockwise)
        s_and_a_layout.addWidget(rotate_robot_anticlockwise)
        s_and_a_layout.addWidget(automate_motion)
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

        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Put them together, quit button on the bottom
        mid_layout = QVBoxLayout()

        mid_layout.addWidget(self.robot_scene)
        mid_layout.addWidget(quit_button)

        return mid_layout

    # Right side sliders/buttons (Gaussian/Kalman filtering)
    # def _init_right_layout_(self):
    #
    #     # The parameters of the robot we're simulating (world/tree/robot state)
    #     parameters = QGroupBox('World parameters, use update button to set')
    #     parameter_layout = QVBoxLayout()
    #     self.prob_see_tree_if_tree = SliderFloatDisplay('Prob see tree if tree', 0.01, 0.99, 0.8)
    #     self.prob_see_tree_if_not_tree = SliderFloatDisplay('Prob see tree if not tree', 0.01, 0.99, 0.1)
    #     self.prob_move_left_if_left = SliderFloatDisplay('Prob move left if left', 0.1, 0.85, 0.8)
    #     self.prob_move_right_if_left = SliderFloatDisplay('Prob move right if left', 0.0, 0.1, 0.05)
    #     self.prob_move_right_if_right = SliderFloatDisplay('Prob move right if right', 0.1, 0.85, 0.8)
    #     self.prob_move_left_if_right = SliderFloatDisplay('Prob move left if right', 0.0, 0.1, 0.05)
    #     self.prob_move_up_if_up = SliderFloatDisplay('Prob move up if up', 0.1, 0.85, 0.8)
    #     self.prob_move_down_if_up = SliderFloatDisplay('Prob move down if up', 0.0, 0.1, 0.05)
    #     self.prob_move_down_if_down = SliderFloatDisplay('Prob move down if down', 0.1, 0.85, 0.8)
    #     self.prob_move_up_if_down = SliderFloatDisplay('Prob move up if down', 0.0, 0.1, 0.05)
    #     self.prob_rotate_robot = SliderFloatDisplay('Prob move up if down', 0.0, 0.1, 0.05)
    #
    #     # update_world_state = QPushButton('Update world state')
    #     # update_world_state.clicked.connect(self.set_probabilities)
    #
    #     parameter_layout.addWidget(self.prob_see_tree_if_tree)
    #     parameter_layout.addWidget(self.prob_see_tree_if_not_tree)
    #     parameter_layout.addWidget(self.prob_move_left_if_left)
    #     parameter_layout.addWidget(self.prob_move_right_if_left)
    #     parameter_layout.addWidget(self.prob_move_right_if_right)
    #     parameter_layout.addWidget(self.prob_move_left_if_right)
    #     parameter_layout.addWidget(self.prob_move_up_if_up)
    #     parameter_layout.addWidget(self.prob_move_down_if_up)
    #     parameter_layout.addWidget(self.prob_move_down_if_down)
    #     parameter_layout.addWidget(self.prob_move_up_if_down)
    #     parameter_layout.addWidget(self.prob_rotate_robot)
    #
    #     # parameter_layout.addWidget(update_world_state)
    #
    #     parameters.setLayout(parameter_layout)
    #
    #     right_side_layout = QVBoxLayout()
    #     right_side_layout.addWidget(parameters)
    #     right_side_layout.addStretch()
    #
    #     return right_side_layout

    def get_tree_on_orchard(self, sense):
        sense_mat = np.matrix([[sense[0]], [sense[1]], [1]])
        translation_x = self.robot_scene.robot.robot_x
        translation_y = self.robot_scene.robot.robot_y
        translation_matrix = np.matrix([[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]])
        trees_translation = np.matmul(translation_matrix, sense_mat)
        rotation_matrix = np.matrix([[np.cos(np.radians(self.robot_scene.robot.robot_angle)),
                                      np.sin(np.radians(self.robot_scene.robot.robot_angle)), 0],
                                     [-np.sin(np.radians(self.robot_scene.robot.robot_angle)),
                                      np.cos(np.radians(self.robot_scene.robot.robot_angle)), 0],
                                     [0, 0, 1]])
        trees_rotated = np.matmul(rotation_matrix, trees_translation)
        min = 1000
        tree_loc = trees_rotated
        tree_matrix_transpose = np.transpose(self.robot_scene.tree_matrix)
        for tree in tree_matrix_transpose:
            d = np.sqrt(
                (tree.item(0) - trees_rotated.item(0)) ** 2 + (tree.item(1) - trees_rotated.item(1)) ** 2)
            if d < min:
                min = d
                tree_loc = tree
                if d == 0:
                    return tree_loc.item(0), tree_loc.item(1)
        return tree_loc.item(0), tree_loc.item(1)

    def get_tree_on_orchard_estimate(self, sense):
        sense_mat = np.matrix([[sense[0]], [sense[1]], [1]])
        translation_x = self.robot_scene.robot_estimation.robot_est_x
        translation_y = self.robot_scene.robot_estimation.robot_est_y
        translation_matrix = np.matrix([[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]])
        trees_translation = np.matmul(translation_matrix, sense_mat)
        rotation_matrix = np.matrix([[np.cos(np.radians(self.robot_scene.robot.robot_angle)),
                                      np.sin(np.radians(self.robot_scene.robot.robot_angle)), 0],
                                     [-np.sin(np.radians(self.robot_scene.robot.robot_angle)),
                                      np.cos(np.radians(self.robot_scene.robot.robot_angle)), 0],
                                     [0, 0, 1]])
        trees_rotated = np.matmul(rotation_matrix, trees_translation)
        min = 1000
        tree_loc = trees_rotated
        tree_matrix_transpose = np.transpose(self.robot_scene.tree_matrix)
        tree_ind = 0
        min_tree_ind = 0
        for tree in tree_matrix_transpose:
            d = np.sqrt(
                (tree.item(0) - trees_rotated.item(0)) ** 2 + (tree.item(1) - trees_rotated.item(1)) ** 2)
            if d < min:
                min = d
                tree_loc = tree
                min_tree_ind = tree_ind
                if d == 0:
                    return tree_loc.item(0), tree_loc.item(1)
            tree_ind = tree_ind + 1

        left_neighbour = [tree_loc.item(0), tree_loc.item(1)]
        right_neighbour = [tree_loc.item(0), tree_loc.item(1)]
        if min_tree_ind - 1 >= 0 :
            if abs(tree_matrix_transpose.item(((min_tree_ind - 1) * 3)) - tree_loc.item(0)) < 30:
                left_neighbour = [tree_matrix_transpose.item(((min_tree_ind - 1) * 3)),
                          tree_matrix_transpose.item(((min_tree_ind - 1) * 3) + 1)]
        if min_tree_ind + 1 < tree_ind :
            if abs(tree_matrix_transpose.item(((min_tree_ind + 1) * 3)) - tree_loc.item(0)) < 30:
                right_neighbour = [tree_matrix_transpose.item(((min_tree_ind + 1) * 3)),
                           tree_matrix_transpose.item(((min_tree_ind + 1) * 3) + 1)]
        return [[tree_loc.item(0), tree_loc.item(1)], right_neighbour, left_neighbour]

    def automate_motion(self):
        tree_list = []
        while (
                self.robot_scene.orchard.orchard_startx + self.robot_scene.orchard.orchard_width) > self.robot_scene.robot.robot_x > self.robot_scene.orchard.orchard_startx and self.robot_scene.orchard.orchard_starty < self.robot_scene.robot.robot_y < (
                self.robot_scene.orchard.orchard_starty + self.robot_scene.orchard.orchard_length):
            sense = self.query_tree_sensor()
            while sense is None:
                self.rotate_clockwise()
                sense = self.query_tree_sensor()
                if sense:
                    break
                if self.robot_scene.robot.robot_angle >= 345:
                    move_dirn = np.random.randint(1, 4)
                    if move_dirn == 1:
                        for i in range(5):
                            self.move_up()
                    elif move_dirn == 2:
                        for i in range(5):
                            self.move_right()
                    elif move_dirn == 3:
                        for i in range(5):
                            self.move_down()
                    elif move_dirn == 3:
                        for i in range(5):
                            self.move_left()
                    self.robot_scene.robot.robot_angle = 0
            tree_loc = self.get_tree_on_orchard(sense)
            self.robot_scene.tree_detected = list(tree_loc)
            if tree_loc not in tree_list:
                tree_list.append(tree_loc)

            if len(tree_list) == 30:
                break
            self.move_right()
            self.query_tree_sensor()
        self.robot_scene.draw_plot()

    def move_left(self):
        move_dist = np.random.normal(0, 1)
        if self.robot_scene.robot.robot_x + move_dist < self.robot_scene.orchard.orchard_startx + self.robot_scene.orchard.orchard_width:
            self.robot_scene.robot.update_robot(-move_dist, 0, 0)
            for p in self.robot_scene.particles:
                move_flag = np.random.normal(0, 1)
                p.move_by(-move_flag, 0)
        self.repaint()

    def move_right(self):

        # Ground Truth
        move_dist = 10 + np.random.normal(0, 1)
        if self.robot_scene.robot.robot_x + move_dist < self.robot_scene.orchard.orchard_startx + self.robot_scene.orchard.orchard_width:
            self.robot_scene.robot.update_robot(move_dist, 0, 0)

            # Particle
            est_move = 0
            for p in self.robot_scene.particles:
                move_flag = 10 + np.random.normal(0, 1)
                est_move += move_flag
                p.move_by(move_flag, 0)
            mean_est_move = est_move / 1000
            self.robot_scene.robot_estimation.update_robot(mean_est_move, 0, 0)

        self.repaint()

    def move_up(self):
        move_dist = 10 + np.random.normal(0, 1)
        if self.robot_scene.robot.robot_x + move_dist < self.robot_scene.orchard.orchard_startx + self.robot_scene.orchard.orchard_width:
            self.robot_scene.robot.update_robot(0, -move_dist, 0)
            for p in self.robot_scene.particles:
                move_flag = 10 + np.random.normal(0, 1)
                p.move_by(0, -move_flag)
        self.repaint()

    def move_down(self):
        move_dist = 10 + np.random.normal(0, 1)
        if self.robot_scene.robot.robot_x + move_dist < self.robot_scene.orchard.orchard_startx + self.robot_scene.orchard.orchard_width:
            self.robot_scene.robot.update_robot(0, move_dist, 0)
            for p in self.robot_scene.particles:
                move_flag = 10 + np.random.normal(0, 1)
                p.move_by(0, move_flag)
        self.repaint()

    def rotate_clockwise(self):
        self.robot_scene.robot_estimation.update_robot(0, 0, +10)
        move_angle = 0.5 + np.random.normal(0.5, 0.5)
        self.robot_scene.robot.update_robot(0, 0, move_angle)
        self.repaint()

    def rotate_anticlockwise(self):
        self.robot_scene.robot_estimation.update_robot(0, 0, -10)
        move_angle = 0.5 + np.random.normal(0.5, 0.5)
        self.robot_scene.robot.update_robot(0, 0, -move_angle)

        self.repaint()

    def query_tree_sensor(self):
        tree_loc_x, tree_loc_y, sensed_trees = Particle.sensor_query(self.robot_scene.robot,
                                                                     self.robot_scene.tree_matrix,
                                                                     self.robot_scene.pointer_length)

        tree_loc = tree_loc_x, tree_loc_y
        # ground truth
        tree_from_orchard = self.get_tree_on_orchard(tree_loc)

        # estimated tree location
        tree_estimate = self.get_tree_on_orchard_estimate(tree_loc)

        robot_tree_dist = np.sqrt(np.power(tree_loc_x, 2) + np.power(tree_loc_y, 2))

        self.robot_scene.robot_sensed_trees = []
        for t in sensed_trees:
            sense = t.item(0), t.item(1)
            self.robot_scene.robot_sensed_trees.append(self.get_tree_on_orchard(sense))
        self.robot_scene.tree_detected = list(tree_from_orchard)
        self.robot_scene.sensor_text = "(" + str(tree_loc) + ")"

        if tree_loc:
            self.robot_scene.particles, self.robot_scene.robot_estimation.robot_est_x, self.robot_scene.robot_estimation.robot_est_y, self.robot_scene.prev_robot_tree_dist = Particle.particle_filter(
                robot_tree_dist, tree_estimate, self.robot_scene.particles, self.robot_scene.trees,
                self.robot_scene.robot_estimation,
                self.robot_scene.orchard, self.robot_scene.prev_robot_tree_dist)

        tree_estimate_2 = self.get_tree_on_orchard_estimate(tree_loc)
        self.robot_scene.tree_detected_estimate = tree_estimate_2[0]

        # self.robot_scene.check_robot_estimation()

        self.repaint()
        return tree_loc

    def draw(self, _):
        self.robot_scene.draw()


if __name__ == '__main__':
    app = QApplication([])
    gui = StateEstimationGUI()
    gui.show()
    app.exec_()
