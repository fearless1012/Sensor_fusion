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
import Robot_path_from_IMU as path
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
        self.width = 1400
        self.height = 900

        # State/action text
        self.sensor_text = "No sensor"
        self.action_text = "No action"
        self.move_text = "No move"
        self.loc_text = "No location"

        # The world state
        orchardWidth = 800
        orchardLength = 800
        orchard_startx = (self.width - orchardWidth) / 2
        orchard_starty = 0
        self.orchard = Orchard.Orchard(orchard_startx, orchard_starty, orchardWidth, orchardLength)
        self.trees, self.tree_count, self.tree_matrix = self.orchard.add_trees()

        # Robot Initialization
        robot_sensorRange = 196
        robot_sensorFOV = 42.5  # in degrees
        robot_startx = orchard_startx + 10
        robot_starty = orchard_starty + 320
        self.robot = Robot.Robot(robot_startx, robot_starty, robot_sensorRange, robot_sensorFOV)

        # Particle Initialize
        self.estx, self.esty = self.robot.return_estimation()
        self.particles = Particle.initialize_particles(self.estx, self.esty, self.robot.heading_est)

        # Highlight detected tree
        self.tree_detected = None

        # Convergence plot
        self.convergence = []

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

        # Draw the Trees

        pen = QPen(Qt.darkGreen, 4, Qt.SolidLine)
        qp.setPen(pen)
        for j in range(self.tree_count - 1):
            qp.drawPoint(self.trees[j].tree_x,
                         self.trees[j].tree_y)

        pen = QPen(Qt.darkBlue, 4, Qt.SolidLine)
        qp.setPen(pen)
        for j in range(self.tree_count - 1):
            qp.drawPoint(self.tree_matrix.item(0),
                         self.tree_matrix.item(1))

        # pen = QPen(Qt.darkYellow, 4, Qt.SolidLine)
        # qp.setPen(pen)
        #
        #
        #
        if self.tree_detected:
            pen = QPen(Qt.darkRed, 4, Qt.SolidLine)
            qp.setPen(pen)
            for t in self.tree_detected:
                qp.drawPoint(t[0], t[1])

    def draw_robot(self, qp, path):
        # Draw the sensor range first
        # The Intel RealSense depth camera D415 has an approximately 65ۜ° field of view (FOV). ange 0.3m to over 10m. Noise ≤ 2%
        # The Intel RealSense D435 depth camera has a wider FOV at approximately 85°. Range 0.2m to over 10m
        # For T265 : 173ۜ° Diagonal field of view (FOV).

        pen = QPen(Qt.yellow, 2, Qt.SolidLine)
        qp.setPen(pen)

        # cos_val_sensor = np.cos(np.radians(self.robot.heading - 90))
        # sin_val_sensor = np.sin(np.radians(self.robot.heading - 90))
        # x_sensor = self.robot.x + (self.robot.sensorRange * cos_val_sensor)
        # y_sensor = self.robot.y + (self.robot.sensorRange * sin_val_sensor)

        cos_val_up = np.cos(np.radians(self.robot.heading - self.robot.sensorFOV - 90))
        sin_val_up = np.sin(np.radians(self.robot.heading - self.robot.sensorFOV - 90))
        cos_val_down = np.cos(np.radians(self.robot.heading + self.robot.sensorFOV - 90))
        sin_val_down = np.sin(np.radians(self.robot.heading + self.robot.sensorFOV - 90))

        # x_p_up = self.robot.x + (self.robot.sensorRange * cos_val_up)
        # y_p_up = self.robot.y + (self.robot.sensorRange * sin_val_up)
        # x_p_down = self.robot.x + (self.robot.sensorRange * cos_val_down)
        # y_p_down = self.robot.y + (self.robot.sensorRange * sin_val_down)

        qp.drawLine(self.robot.x, self.robot.y, self.robot.x_p_up, self.robot.y_p_up)
        qp.drawLine(self.robot.x, self.robot.y, self.robot.x_p_down, self.robot.y_p_down)
        path.moveTo(self.robot.x_p_up, self.robot.y_p_up)
        path.cubicTo(self.robot.x_p_up, self.robot.y_p_up, self.robot.x_sensor, self.robot.y_sensor,
                     self.robot.x_p_down, self.robot.y_p_down)
        qp.drawPath(path)
        # arc_rect = QRect(self.robot.x - self.robot.sensorRange, self.robot.y - self.robot.sensorRange,
        #                  self.robot.sensorRange, self.robot.sensorRange)
        # qp.drawArc(arc_rect, (-self.robot.heading) * 16, +self.robot.sensorFOV * 16 )
        # qp.drawArc(arc_rect, (-self.robot.heading) * 16, -self.robot.sensorFOV * 16)

        # Draw robot body as a point
        robot_radius = 3
        pen = QPen(Qt.darkMagenta, 1, Qt.SolidLine)
        qp.setPen(pen)
        qp.setBrush(Qt.darkMagenta)
        center = QPoint(self.robot.x, self.robot.y)
        qp.drawEllipse(center, robot_radius, robot_radius)

        # Draw robot header
        pen = QPen(Qt.darkMagenta, 1, Qt.SolidLine)
        qp.setPen(pen)
        cos_val = np.cos(np.radians(self.robot.heading))
        sin_val = np.sin(np.radians(self.robot.heading))
        x_h = self.robot.x + (2 * robot_radius * cos_val)
        y_h = self.robot.y + (2 * robot_radius * sin_val)
        x_h_up = self.robot.x + (robot_radius * cos_val_up)
        y_h_up = self.robot.y + (robot_radius * sin_val_up)
        x_h_down = self.robot.x + (robot_radius * cos_val_down)
        y_h_down = self.robot.y + (robot_radius * sin_val_down)
        qp.drawLine(x_h, y_h, x_h_up, y_h_up)
        qp.drawLine(x_h, y_h, x_h_down, y_h_down)

        pen = QPen(Qt.darkCyan, 4, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawPoint(self.robot.x_pf, self.robot.y_pf)

        pen = QPen(Qt.darkBlue, 4, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawPoint(self.robot.x_est, self.robot.y_est)

        pen = QPen(Qt.darkRed, 2, Qt.SolidLine)
        qp.setPen(pen)
        # text_loc = QPoint(self.orchard.orchard_startx + self.orchard.orchard_width + 10 , self.orchard.orchard_starty + 30)
        text_loc = QPoint(self.orchard.orchard_startx,
                          self.orchard.orchard_starty + self.orchard.orchard_length + 30)
        text_content = "Red - Ground Truth position of Robot" + str(self.robot.x) + str(self.robot.y)
        qp.drawText(text_loc, text_content)

        pen = QPen(Qt.darkBlue, 2, Qt.SolidLine)
        qp.setPen(pen)
        text_loc = QPoint(self.orchard.orchard_startx + self.orchard.orchard_width + 10,
                          self.orchard.orchard_starty + 50)
        qp.drawText(text_loc, "Blue - Dead reckoning estimate of Robot")

        pen = QPen(Qt.darkCyan, 2, Qt.SolidLine)
        qp.setPen(pen)
        text_loc = QPoint(self.orchard.orchard_startx + self.orchard.orchard_width + 10,
                          self.orchard.orchard_starty + 70)
        qp.drawText(text_loc, "Cyan - Particle filter localisation")
        # text_loc = QPoint(20, self.orchard.orchard_starty + self.orchard.orchard_length + 20)
        # qp.drawText(text_loc, self.sensor_text)

        # For convergence plot
        error = np.sqrt((self.robot.x - self.robot.x_pf) ** 2 + (
                self.robot.y - self.robot.y_pf) ** 2)
        self.convergence.append(error)

    def draw_particles(self, qp, path):
        pen = QPen(Qt.lightGray, 3, Qt.SolidLine)
        qp.setPen(pen)
        for particle in self.particles:
            # path.moveTo(particle.x, particle.y)
            # path.addPath(, particle.y)
            qp.drawPoint(particle.x, particle.y)

    def draw_plot(self):
        plt.figure(2)
        x_values = []
        for i in range(len(self.convergence)):
            x_values.append(i)
        plt.plot(x_values, self.convergence)
        plt.title("Convergence Plot for Particle Filter Localisation")
        plt.xlabel("Time step")
        plt.ylabel("Distance from ground truth(scale = y/10 ft)")
        # plt.show()
        plt.savefig("convergence_plot")


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

    # Set up the left set of sliders/buttons (state estimation)
    def _init_left_layout_(self):
        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)

        # Action/sensor buttons - update state estimation
        query_tree_sensor_button = QPushButton('Query tree sensor')
        query_tree_sensor_button.pressed.connect(self.query_tree_sensor)

        self.move_forward_button = QPushButton('Move forward')
        self.move_forward_button.clicked.connect(self.move_forward)

        move_reverse_button = QPushButton('Move reverse')
        move_reverse_button.pressed.connect(self.move_reverse)

        rotate_robot_clockwise = QPushButton('Rotate Clockwise')
        rotate_robot_clockwise.pressed.connect(self.rotate_clockwise)

        rotate_robot_anticlockwise = QPushButton('Rotate Anticlockwise')
        rotate_robot_anticlockwise.pressed.connect(self.rotate_anticlockwise)

        automate_motion = QPushButton('Automate motion')
        automate_motion.clicked.connect(self.automate_motion)

        pf_localization = QPushButton('PF Localization')
        pf_localization.clicked.connect(self.particle_filter_localisation)

        s_and_a = QGroupBox('State estimation: query state and do action')
        s_and_a_layout = QVBoxLayout()
        s_and_a_layout.addWidget(query_tree_sensor_button)
        s_and_a_layout.addWidget(self.move_forward_button)
        s_and_a_layout.addWidget(move_reverse_button)
        s_and_a_layout.addWidget(rotate_robot_clockwise)
        s_and_a_layout.addWidget(rotate_robot_anticlockwise)
        s_and_a_layout.addWidget(automate_motion)
        s_and_a_layout.addWidget(pf_localization)
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

    def robot_to_orchard_reference_frame(self, obj_loc):
        translation_x = self.robot_scene.robot.x
        translation_y = self.robot_scene.robot.y
        translation_matrix = np.matrix([[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]])
        trees_translation = np.matmul(translation_matrix, np.transpose(obj_loc))
        rotation_matrix = np.matrix([[np.cos(np.radians(self.robot_scene.robot.heading)),
                                      np.sin(np.radians(self.robot_scene.robot.heading)), 0],
                                     [-np.sin(np.radians(self.robot_scene.robot.heading)),
                                      np.cos(np.radians(self.robot_scene.robot.heading)), 0],
                                     [0, 0, 1]])
        trees_rotated = np.matmul(rotation_matrix, trees_translation)
        min = 1000
        trees_rotated_transpose = np.transpose(trees_rotated)
        tree_matrix_transpose = np.transpose(self.robot_scene.tree_matrix)

        sensed_tree_matches = []
        for tree_est in trees_rotated_transpose:
            min = 1000
            for tree_ground in tree_matrix_transpose:
                d = np.sqrt(
                    (tree_ground.item(0) - tree_est.item(0)) ** 2 + (tree_ground.item(1) - tree_est.item(1)) ** 2)
                if d < min:
                    min = d
                    matched_tree = tree_ground
                    if d == 0:
                        break

            sensed_tree_matches.append([matched_tree.item(0), matched_tree.item(1)])
        # return sensed_tree_matches
        return trees_rotated_transpose

    # def automate_motion(self):
    #     time = 0.0
    #     tree_list = []
    #     hertz_track = 0
    #
    #     while SIM_TIME >= time and (
    #             self.robot_scene.orchard.orchard_startx + self.robot_scene.orchard.orchard_width) > self.robot_scene.robot.x > self.robot_scene.orchard.orchard_startx and self.robot_scene.orchard.orchard_starty < self.robot_scene.robot.y < self.robot_scene.orchard.orchard_starty + self.robot_scene.orchard.orchard_length:
    #         time += DT
    #         self.move_forward(time)
    #         #30 hz freq of camera
    #         #imu -  60 - 200 hz
    #         #change sensor to imu and cameras reading
    #         if hertz_track % 3 == 0:
    #             self.particle_filter_localisation()
    #
    #         hertz_track += 1
    #
    #     self.robot_scene.draw_plot()

    def get_imu_path(self):
        path = pd.read_csv('gps_path_heading.csv')
        # x,y = path['x'], path['y']
        return path

    def automate_motion(self):
        sim_i = 0
        tree_list = []
        hertz_track = 0
        path = self.get_imu_path()

        while sim_i < len(path):
            sim_i += 1
            # if path.loc[sim_i, 'type'] == 'gyr':
            #     self.robot_scene.robot.set_heading(-path.loc[sim_i, 'heading_store'])
            # else:
            #     self.robot_scene.robot.move_dd(path.loc[sim_i, 'heading_store']*10)
            self.robot_scene.robot.set_heading(np.pi/4 - path.loc[sim_i-1, 'heading'])
            self.robot_scene.robot.move_dd(path.loc[sim_i-1, 'dd']*10)
            if sim_i % 10 == 0:
                self.repaint()
            # if hertz_track % 3 == 0:
            #     self.particle_filter_localisation()

    def move_forward(self, time=None):
        delta_xy = self.robot_scene.robot.move_forward(self.robot_scene.orchard, time, DT)
        self.particle_filter_spread(delta_xy)
        self.repaint()

    def move_reverse(self):
        delta_xy = self.robot_scene.robot.move_reverse(self.robot_scene.orchard)
        self.particle_filter_spread(delta_xy)
        self.repaint()

    def rotate_clockwise(self):
        self.robot_scene.robot.rotate_clockwise(self.robot_scene.orchard)
        self.repaint()

    def rotate_anticlockwise(self):
        self.robot_scene.robot.rotate_anticlockwise(self.robot_scene.orchard)
        self.repaint()

    def query_tree_sensor(self):
        tree_loc, sensed_trees, tree_distances, tree_directions = self.robot_scene.robot.sensor_query(
            self.robot_scene.tree_matrix)
        if sensed_trees:
            # min_tree_dist = min(tree_distances)
            # i = 0
            # n = len(sensed_trees)
            # while i < n:
            #     if tree_distances[i] > min_tree_dist + 20:
            #         tree_distances.pop(i)
            #         sensed_trees.pop(i)
            #         tree_directions.pop(i)
            #         n = n-1
            #     else:
            #         i += 1
            tree_from_orchard = sensed_trees  # self.robot_to_orchard_reference_frame(sensed_trees)
            self.robot_scene.tree_detected = tree_from_orchard
            self.robot_scene.sensor_text = "(" + str(tree_loc) + ")"
            self.repaint()
            return tree_from_orchard, tree_distances
        else:
            print("heres")
            return None

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
