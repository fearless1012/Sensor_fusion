import math
from datetime import datetime

import numpy as np
import pandas as pd
import utm as utm
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt, animation
import numbers

from pandas.core.dtypes.common import is_numeric_dtype

plt.style.use('dark_background')


def getData(fname):
    df = pd.read_csv(fname)
    return df


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
    plt.ylabel('z')
    plt.savefig('imu.png')


if __name__ == '__main__':
    # acc_path_data = pd.DataFrame(
    #     columns=['field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
    #              'field.angular_velocity.z',
    #              'field.linear_acceleration.x', 'field.linear_acceleration.y',
    #              'field.linear_acceleration.z', 'file_name'])
    # gps_path_data = pd.DataFrame(columns=['time', 'field.header.stamp', 'North', 'West', 'file_name'])
    # gyro_path_data = pd.DataFrame(
    #     columns=['field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
    #              'field.angular_velocity.z',
    #              'field.linear_acceleration.x', 'field.linear_acceleration.y',
    #              'field.linear_acceleration.z', 'file_name'])
    # imu_path_data = pd.DataFrame(
    #     columns=['field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
    #              'field.angular_velocity.z',
    #              'field.linear_acceleration.x', 'field.linear_acceleration.y',
    #              'field.linear_acceleration.z', 'file_name', 'type'])
    # for i in range(11):
    #     file_name = "./Data/acc_" + str(i) + ".csv"
    #     acc = getData(file_name)
    #     acc['file_name'] = file_name
    #     acc['type'] = "acc"
    #
    #     gps_filename = "./Data/gps_" + str(i) + ".csv"
    #     gps = pd.read_csv(gps_filename, skiprows=range(1))
    #     gps.columns = ['%time', 'field.header.seq', 'field.header.stamp',
    #                    'field.header.frame_id', 'field.sentence', 'rand', 'North', 'N-label', 'West', 'W-label',
    #                    'rand_1', 'rand_2', 'rand_3', 'rand_4', 'rand_5', 'rand_6', 'rand_7', 'rand_8', 'rand_9']
    #     gps['file_name'] = gps_filename
    #
    #     gyro_filename = "./Data/gyro_" + str(i) + ".csv"
    #     gyro = getData(gyro_filename)
    #     gyro['file_name'] = gyro_filename
    #     gyro['type'] = "gyr"
    #
    #     acc_data = acc[['field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
    #                     'field.angular_velocity.z',
    #                     'field.linear_acceleration.x', 'field.linear_acceleration.y',
    #                     'field.linear_acceleration.z', 'file_name', 'type']]
    #     gyro_data = gyro[
    #         ['field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
    #          'field.angular_velocity.z',
    #          'field.linear_acceleration.x', 'field.linear_acceleration.y',
    #          'field.linear_acceleration.z', 'file_name', 'type']]
    #
    #     gps_data = gps[['field.header.stamp', 'North', 'West', 'file_name']]
    #
    #     acc_path_data = acc_path_data.append(acc_data, ignore_index=True)
    #     gps_path_data = gps_path_data.append(gps_data, ignore_index=True)
    #     gyro_path_data = gyro_path_data.append(gyro_data, ignore_index=True)
    #
    # acc_path_data['field.linear_acceleration.x'] = acc_path_data['field.linear_acceleration.x'] * 100
    # acc_path_data['field.linear_acceleration.z'] = acc_path_data['field.linear_acceleration.z'] * 100
    # acc_path_data['delta_time'] = acc_path_data['field.header.stamp'].diff() / 1000000000
    # acc_path_data.loc[0, 'delta_time'] = 0
    # acc_path_data['delta_x'] = acc_path_data['field.linear_acceleration.x'] * acc_path_data['delta_time'] * \
    #                            acc_path_data['delta_time']
    # acc_path_data['delta_z'] = acc_path_data['field.linear_acceleration.z'] * acc_path_data['delta_time'] * \
    #                            acc_path_data['delta_time']
    #
    # gyro_path_data['delta_time'] = gyro_path_data['field.header.stamp'].diff() / 1000000000
    # gyro_path_data.loc[0, 'delta_time'] = 0
    # gyro_path_data['delta_angle'] = gyro_path_data['field.angular_velocity.y'] * gyro_path_data['delta_time']
    #
    # imu_path_data = pd.concat([acc_path_data, gyro_path_data], axis=0, ignore_index=True).sort_values('field.header'
    #                                                                                                   '.stamp',
    #                                                                                                   ignore_index=True)
    # x = []
    # y = []
    # x.append(0)
    # y.append(0)
    # heading = 0
    # j = 1
    # total_d = 0
    # imu_path_data = imu_path_data[imu_path_data['field.header.stamp'] > gps_path_data.loc[0, 'field.header.stamp']]
    # imu_path_data.reset_index(drop=True, inplace=True)
    # imu_path_data['heading_store'] = np.zeros(len(imu_path_data))
    # for i in range(1, len(imu_path_data)):
    #     if imu_path_data.loc[i, 'type'] == 'gyr':
    #         if math.isnan(imu_path_data.loc[i, 'delta_angle']):
    #             imu_path_data.loc[i, 'delta_angle'] = 0
    #         heading = heading - imu_path_data.loc[i, 'delta_angle']
    #         imu_path_data.loc[i, 'heading_store'] = heading
    #     else:
    #         if math.isnan(imu_path_data.loc[i, 'delta_x']) or math.isnan(imu_path_data.loc[i, 'delta_z']):
    #             imu_path_data.loc[i, 'delta_x'] = 0
    #             imu_path_data.loc[i, 'delta_z'] = 0
    #         delta_d = np.sqrt(imu_path_data.loc[i, 'delta_x'] ** 2 + imu_path_data.loc[i, 'delta_z'] ** 2)
    #         total_d += delta_d
    #         imu_path_data.loc[i, 'heading_store'] = delta_d
    #         x.append(x[j - 1] + (delta_d * np.cos(heading)))
    #         y.append(y[j - 1] + (delta_d * np.sin(heading)))
    #         j += 1
            # if total_d > 20:
            #     print(imu_path_data.loc[i, 'field.header.stamp'])
            #     break

    # print(total_d)
    x = []
    y = []
    x.append(0)
    y.append(0)
    heading = 0
    j = 1
    imu_filename = pd.read_csv('imu_path_heading.csv')
    path = pd.read_csv(imu_filename, skiprows=range(1))
    sim_i = 1
    while sim_i < len(path):
        if path.loc[sim_i, 'type'] == 'gyr':
            heading = path.loc[sim_i, 'heading_store']
        else:
            x = path.loc[sim_i, 'heading_store']
    plotPaths(x, y, 1)
    plt.show()
    d = {'x':x,'y':y}
    imu_path = pd.DataFrame(d)
    imu_path = imu_path_data[['type', 'heading_store', 'field.header.stamp']]
    # print(imu_path)
    # imu_path.to_csv("imu_path_heading.csv")

    gps_path_data['lat'] = np.deg2rad(gps_path_data['North'])
    gps_path_data['lon'] = np.deg2rad(gps_path_data['West'])
    gps_path_data['lon'] = gps_path_data['lon'] - 180
    gps_path_data['utm_data_x'] = ''
    gps_path_data['utm_data_y'] = ''
    for i in range(len(gps_path_data)):
        utm_data = utm.from_latlon(gps_path_data.loc[i, 'lat'], gps_path_data.loc[i, 'lon'])
        gps_path_data.loc[i, 'utm_data_x'] = utm_data[0]
        gps_path_data.loc[i, 'utm_data_y'] = utm_data[1]

    gps_x = []
    gps_y = []
    gps_x.append(0)
    gps_y.append(0)
    td = 0
    gps_dd = []
    gps_heading = []
    time_stamp = []
    for i in range(1, len(gps_path_data)):
        dx_dash = gps_path_data.loc[i, 'utm_data_x'] - gps_path_data.loc[i - 1, 'utm_data_x']
        dy_dash = gps_path_data.loc[i, 'utm_data_y'] - gps_path_data.loc[i - 1, 'utm_data_y']

        dy_dash = dy_dash
        dx_dash = dx_dash

        # dx = gps_x[i - 1] - dx_dash
        # dy = dy_dash + gps_y[i - 1]
        # gps_x.append(dx)
        # gps_y.append(dy)

        dd = np.sqrt(dx_dash ** 2 + dy_dash ** 2)
        d_theta = np.pi - np.arctan2(dy_dash, dx_dash)
        td = td + dd
        dd = dd
        gps_dd.append(dd)
        gps_heading.append(d_theta)
        time_stamp.append(gps_path_data.loc[i, 'field.header.stamp'])
        gps_x.append(gps_x[i - 1] + (dd * np.cos(d_theta)))
        gps_y.append(gps_y[i - 1] + (dd * np.sin(d_theta)))

    print(td)
    plotPaths(gps_x, gps_y, 1)
    plt.show()
    # d = {'x':gps_x,'y':gps_y}
    print(gps_path_data.columns)
    d = {'heading': gps_heading, 'dd': gps_dd, 'field.header.stamp':time_stamp}
    gps_path = pd.DataFrame(d)
    gps_path.to_csv("gps_path_heading.csv")

    gps_imu_comparison = pd.concat([imu_path, gps_path], axis=0, ignore_index=True).sort_values('field.header.stamp', ignore_index=True)
    gps_imu_comparison.to_csv("gps_imu_comparison.csv")
    print("here")
