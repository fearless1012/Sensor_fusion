# import math

import numpy as np


class Robot:
    def __init__(self, robot_startx, robot_starty, robot_sensorRange, robot_sensorFOV):
        self.x = robot_startx
        self.y = robot_starty
        self.heading = 0
        self.start_speed = 10
        self.acceleration = 0

        # Estimation Variables
        self.x_est = robot_startx
        self.y_est = robot_starty
        self.heading_est = 0

        # localization Variables
        self.x_pf = robot_startx
        self.y_pf = robot_starty
        self.heading_pf = 0

        # Sensor Variables
        self.sensorRange = robot_sensorRange
        self.sensorFOV = robot_sensorFOV
        self.x_sensor = self.x + (self.sensorRange * np.cos(np.radians(self.heading - 90)))
        self.y_sensor = self.y + (self.sensorRange * np.sin(np.radians(self.heading - 90)))
        self.x_p_up = self.x + (self.sensorRange * np.cos(np.radians(self.heading - self.sensorFOV - 90)))
        self.y_p_up = self.y + (self.sensorRange * np.sin(np.radians(self.heading - self.sensorFOV - 90)))
        self.x_p_down = self.x + (self.sensorRange * np.cos(np.radians(self.heading + self.sensorFOV - 90)))
        self.y_p_down = self.y + (self.sensorRange * np.sin(np.radians(self.heading + self.sensorFOV - 90)))

    def move_forward(self, orchard, time=None, DT=None):
        if time:
            move_dist = ((self.start_speed * time + 0.5 * self.acceleration * time * time) - (
                    self.start_speed * (time - DT) + 0.5 * self.acceleration * (time - DT) * (time - DT)))
        else:
            move_dist = 10

        delta_x_est = move_dist * np.cos(np.radians(self.heading))
        delta_y_est = move_dist * np.sin(np.radians(self.heading))
        noise = np.abs(np.random.normal(0, 0.1))
        delta_x = (move_dist + noise) * np.cos(np.radians(self.heading))
        delta_y = (move_dist + noise) * np.sin(np.radians(self.heading))

        if orchard.orchard_startx < self.x + delta_x < orchard.orchard_startx + orchard.orchard_width and orchard.orchard_starty < self.y + delta_y < orchard.orchard_starty + orchard.orchard_length:
            self.update_robot(delta_x, delta_y, 0)
            self.update_robot_est(delta_x_est, delta_y_est, 0)
            return [delta_x_est, delta_y_est]
        return [0, 0]

    def move_reverse(self, orchard, time=None, DT=None):
        if time:
            move_dist = -((self.start_speed * time + 0.5 * self.acceleration * time * time) - (
                    self.start_speed * (time - DT) + 0.5 * self.acceleration * (time - DT) * (time - DT)))
        else:
            move_dist = -10
        delta_x_est = move_dist * np.cos(np.radians(self.heading))
        delta_y_est = move_dist * np.sin(np.radians(self.heading))
        delta_x = (move_dist + np.abs(np.random.normal(0, 0.1))) * np.cos(np.radians(self.heading))
        delta_y = (move_dist + np.abs(np.random.normal(0, 0.1))) * np.sin(np.radians(self.heading))
        if orchard.orchard_startx < self.x + delta_x < orchard.orchard_startx + orchard.orchard_width and orchard.orchard_starty < self.y + delta_y < orchard.orchard_starty + orchard.orchard_length:
            self.update_robot(delta_x, delta_y, 0)
            self.update_robot_est(delta_x_est, delta_y_est, 0)
            return [delta_x_est, delta_y_est]
        return [0, 0]

    def rotate_clockwise(self, orchard):
        move_angle = 10 + np.abs(np.random.normal(0, 5))
        self.update_robot(0, 0, move_angle)
        self.update_robot_est(0, 0, 10)

    def rotate_anticlockwise(self, orchard):
        move_angle = 10 + np.abs(np.random.normal(0, 5))
        self.update_robot(0, 0, -move_angle)

    def update_robot(self, change_x, change_y, change_theta):
        self.x = self.x + change_x
        self.y = self.y + change_y
        self.heading = self.heading + change_theta
        if self.heading >= 360:
            self.heading = self.heading - 360
        if self.heading < 0:
            self.heading = 360 + self.heading
        self.update_sensor_position()

    def update_robot_est(self, change_x, change_y, change_theta):
        self.x_est = self.x_est + change_x
        self.y_est = self.y_est + change_y
        self.heading_est = self.heading_est + change_theta
        if self.heading_est >= 360:
            self.heading_est = self.heading_est - 360
        if self.heading_est < 0:
            self.heading_est = 360 + self.heading_est

    def update_sensor_position(self):
        cos_val_sensor = np.cos(np.radians(self.heading - 90))
        sin_val_sensor = np.sin(np.radians(self.heading - 90))
        cos_val_up = np.cos(np.radians(self.heading - self.sensorFOV - 90))
        sin_val_up = np.sin(np.radians(self.heading - self.sensorFOV - 90))
        cos_val_down = np.cos(np.radians(self.heading + self.sensorFOV - 90))
        sin_val_down = np.sin(np.radians(self.heading + self.sensorFOV - 90))

        self.x_sensor = self.x + (self.sensorRange * cos_val_sensor)
        self.y_sensor = self.y + (self.sensorRange * sin_val_sensor)
        self.x_p_up = self.x + (self.sensorRange * cos_val_up)
        self.y_p_up = self.y + (self.sensorRange * sin_val_up)
        self.x_p_down = self.x + (self.sensorRange * cos_val_down)
        self.y_p_down = self.y + (self.sensorRange * sin_val_down)

    def set_positions(self, x, y, ):
        self.x = x
        self.y = y

    def set_heading(self, heading):
        self.heading = heading
        self.update_sensor_position()

    def move_dd(self, dd):
        self.x += dd * np.cos(self.heading)
        self.y += dd * np.sin(self.heading)
        

    def orchard_to_robot_reference_frame(self, obj_loc):
        # translation
        # print(self.x, self.y)
        translation_matrix = np.matrix([[1, 0, -self.x], [0, 1, -self.y], [0, 0, 1]])
        trees_translation = np.matmul(translation_matrix, obj_loc)
        # rotation
        # print(self.heading)
        rotation_matrix = np.matrix([[np.cos(np.radians(self.heading)),
                                      -np.sin(np.radians(self.heading)), 0],
                                     [np.sin(np.radians(self.heading)),
                                      np.cos(np.radians(self.heading)), 0],
                                     [0, 0, 1]])
        trees_rotated = np.matmul(rotation_matrix, trees_translation)
        return trees_rotated

    def return_estimated_location(self):
        return [self.x_est, self.y_est]

    def IsTreeDetected(self, x, y):

        # print(x, y, self.x, self.y, self.x_p_up, self.y_p_up, self.x_p_down, self.y_p_down)

        #Another Python code example method
        x1 = self.x
        y1 = self.y
        x2 = self.x_p_up
        y2 = self.y_p_up
        x3 = self.x_p_down
        y3 = self.y_p_down
        xp = x
        yp = y
        c1 = (x2 - x1) * (yp - y1) - (y2 - y1) * (xp - x1)
        c2 = (x3 - x2) * (yp - y2) - (y3 - y2) * (xp - x2)
        c3 = (x1 - x3) * (yp - y3) - (y1 - y3) * (xp - x3)
        if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
            return True
        else:
            return False

        #
        # Area Method
        #sensor_triangle_area = abs(self.x * (self.y_p_up - self.y_p_down) + self.x_p_up * (
        #                 self.y_p_down - self.y) + self.x_p_down * (self.y - self.y_p_up))
        # triangle_1_area = abs(x * (self.y_p_up - self.y_p_down) + self.x_p_up * (
        #         self.y_p_down - y) + self.x_p_down * (y - self.y_p_up))
        # triangle_2_area = abs(self.x * (y - self.y_p_down) + x * (
        #         self.y_p_down - self.y) + self.x_p_down * (self.y - y))
        # triangle_3_area = abs(self.x * (self.y_p_up - y) + self.x_p_up * (
        #         y - self.y) + x * (self.y - self.y_p_up))
        #
        # if triangle_1_area+triangle_2_area+triangle_3_area == sensor_triangle_area:
        #     return True
        # else:
        #     return False


        # Barycentric coordinate system
        # det_T = (self.y_p_up - self.y_p_down) * (self.x - self.x_p_down) - (self.x_p_up - self.x_p_down) * (
        #         self.y - self.y_p_down)
        # alpha = ((self.y_p_up - self.y_p_down) * (x - self.x_p_down) + (self.x_p_down - self.x_p_up) * (
        #         y - self.y_p_down)) / det_T
        # beta = ((self.y_p_down - self.y) * (x - self.x_p_down) + (self.x - self.x_p_down) * (
        #         y - self.y_p_down)) / det_T
        # gamma = 1.0 - alpha - beta
        #
        # if alpha > 0 and beta > 0 and gamma > 0:
        #     return True
        # else:
        #     return False

    def sensor_query(self, tree_matrix):
        print("sensor")
        trees_from_robot = self.orchard_to_robot_reference_frame(tree_matrix)
        # print(tree_matrix)
        # print(trees_from_robot)
        trees_from_robot_transpose = np.transpose(trees_from_robot)
        trees_matrix_transpose = np.transpose(tree_matrix)

        sensed_trees = []
        min_d = 1000
        min_tree = []
        tree_distances = []
        tree_directions = []

        # for tree in trees_from_robot_transpose:
        #     tree_distance = np.sqrt(tree.item(0) ** 2 + tree.item(1) ** 2)
        #     tree_direction = np.arctan2(tree.item(1), tree.item(0))
        #     if tree_distance < self.sensorRange/2 and -self.sensorFOV-90 < np.degrees(tree_direction) < self.sensorFOV-90:
        #         tree_distance_noisy = tree_distance + np.random.normal(0, 0.1)
        #         tree_direction_noisy = tree_direction + np.abs(np.random.normal(0, 1))
        #         estimated_treexy = [tree_distance_noisy * np.cos(tree_direction), tree_distance_noisy * np.sin(tree_direction), 1]
        #         sensed_trees.append(estimated_treexy)
        #         tree_distances.append(tree_distance_noisy)
        #         tree_directions.append(tree_direction_noisy)
        #         if tree_distance_noisy < min_d:
        #             min_d = tree_distance
        #             min_tree = [tree.item(0), tree.item(1)]
        # ============================================================================================================
        #
        for tree in trees_matrix_transpose:


            # print(triangle_1_area + triangle_2_area + triangle_3_area, sensor_triangle_area)

            if self.IsTreeDetected(tree.item(0), tree.item(1)):
                print("tree within sensor")
                tree_distance = np.sqrt((tree.item(0) - self.x) ** 2 + (tree.item(1) - self.y) ** 2)
                tree_direction = np.arctan2((tree.item(1) - self.y), (tree.item(0) - self.x))
                tree_distance_noisy = tree_distance + np.random.normal(0, 0.1)
                tree_direction_noisy = tree_direction #+ np.abs(np.random.normal(0, 1))
                estimated_treexy = [tree_distance_noisy * np.cos(tree_direction),
                                    tree_distance_noisy * np.sin(tree_direction), 1]
                estimated_treexy = [tree.item(0), tree.item(1), 1]
                sensed_trees.append(estimated_treexy)
                tree_distances.append(tree_distance_noisy)
                tree_directions.append(tree_direction_noisy)
                if tree_distance_noisy < min_d:
                    min_d = tree_distance
                    min_tree = [tree.item(0), tree.item(1)]

        if min_d < 1000:
            closest_tree_loc = min_tree
        else:
            closest_tree_loc = None

        # print(closest_tree_loc)

        return closest_tree_loc, sensed_trees, tree_distances, tree_directions

    def return_estimation(self):
        return self.x_est, self.y_est
