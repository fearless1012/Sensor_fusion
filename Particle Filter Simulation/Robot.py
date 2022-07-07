# import math

import numpy as np


class Robot:
    def __init__(self, startx, starty, starth):
        """IMU Dead reckoning"""
        self.x = startx
        self.y = starty
        self.heading = starth

        """Ground Truth"""
        self.x_gf = startx
        self.y_gf = starty
        self.heading_gf = starth

        # localization Variables
        self.x_pf = startx
        self.y_pf = starty
        self.heading_pf = starth

        # Previous state Variables

