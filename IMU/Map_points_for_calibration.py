import math

from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def getData(fname):
    df = pd.read_csv(fname)
    return df


if __name__ == '__main__':

    name_list = ['X1', 'X2', 'X3', 'X4', 'Z1', 'Z2', 'Z3', 'Z4', 'y_rot1', 'y_rot2', 'y_rot3']
    for index_1 in range(len(name_list)):
        acc_file_name = "./Calibration/accel_" + name_list[index_1] + ".csv"
        acc = getData(acc_file_name)
        gyr_file_name = "./Calibration/gyro_" + name_list[index_1] + ".csv"
        gyr = getData(gyr_file_name)
        acc['delta_time'] = acc['field.header.stamp'].diff() / 1000000000
        acc['delta_time'].fillna(0)
        acc['delta_x'] = acc['field.linear_acceleration.x'] * acc['delta_time'] * acc['delta_time']
        acc['x'] = acc['delta_x'].cumsum()
        acc['delta_z'] = -acc['field.linear_acceleration.z'] * acc['delta_time'] * acc['delta_time']
        acc['z'] = acc['delta_z'].cumsum()
        acc['type'] = 'acc'

        gyr['delta_time'] = gyr['field.header.stamp'].diff() / 1000000000
        gyr['delta_time'].fillna(0)
        gyr['type'] = 'gyr'
        gyr['delta_angle'] = -gyr['field.angular_velocity.y'] * gyr['delta_time']

        gyr = gyr[['field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
                   'field.angular_velocity.z', 'type', 'delta_angle']]
        acc = acc[['field.header.stamp', 'field.linear_acceleration.x', 'field.linear_acceleration.y',
                   'field.linear_acceleration.z', 'type', 'delta_x', 'delta_z']]

        imu = pd.concat([acc, gyr], axis=0, ignore_index=True).sort_values('field.header.stamp', ignore_index=True)

        x = []
        y = []
        x.append(0)
        y.append(0)
        heading = 0
        j = 1
        for i in range(1, len(imu)):
            if imu.loc[i, 'type'] == 'gyr':
                if math.isnan(imu.loc[i, 'delta_angle']):
                    imu.loc[i, 'delta_angle'] = 0
                heading = heading + imu.loc[i, 'delta_angle']
            else:
                if math.isnan(imu.loc[i, 'delta_x']) or math.isnan(imu.loc[i, 'delta_z']):
                    imu.loc[i, 'delta_x'] = 0
                    imu.loc[i, 'delta_z'] = 0
                delta_d = np.sqrt(imu.loc[i, 'delta_x']**2 + imu.loc[i, 'delta_z']**2)
                x.append(x[j - 1] + (delta_d * np.cos(heading)))
                y.append(y[j - 1] + (delta_d * np.sin(heading)))
                j += 1

        len_x = x[j - 1]
        len_z = y[j - 1]
        length = np.sqrt(len_x ** 2 + len_z ** 2)

        plt.figure(index_1 + 1)
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(name_list[index_1])
        png_name = "./Calibration/imu_plot_" + name_list[index_1] + ".png"
        plt.text(0, 0, length)
        plt.savefig(png_name)
        plt.show()
