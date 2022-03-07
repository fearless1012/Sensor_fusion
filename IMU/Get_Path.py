import math
from datetime import datetime

import numpy as np
import pandas as pd
import utm as utm
from matplotlib import pyplot as plt
import numbers

from pandas.core.dtypes.common import is_numeric_dtype


def getData(fname):
    df = pd.read_csv(fname)
    return df


if __name__ == '__main__':
    acc_path_data = pd.DataFrame(
        columns=['time', 'field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
                 'field.angular_velocity.z',
                 'field.linear_acceleration.x', 'field.linear_acceleration.y',
                 'field.linear_acceleration.z', 'file_name'])
    gps_path_data = pd.DataFrame(columns=['time', 'field.header.stamp', 'North', 'West', 'file_name'])
    gyro_path_data = pd.DataFrame(
        columns=['time', 'field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
                 'field.angular_velocity.z',
                 'field.linear_acceleration.x', 'field.linear_acceleration.y',
                 'field.linear_acceleration.z', 'file_name'])
    imu_path_data = pd.DataFrame(
        columns=['time', 'field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
                 'field.angular_velocity.z',
                 'field.linear_acceleration.x', 'field.linear_acceleration.y',
                 'field.linear_acceleration.z', 'file_name', 'type'])
    for i in range(11):
        file_name = "./Data/acc_" + str(i) + ".csv"
        acc = getData(file_name)
        acc['time'] = pd.to_datetime(acc['field.header.stamp'] / 1000000000, unit='s')
        acc['file_name'] = file_name
        acc['type'] = "acc"

        gps_filename = "./Data/gps_" + str(i) + ".csv"
        gps = pd.read_csv(gps_filename, skiprows=range(1))
        gps.columns = ['%time', 'field.header.seq', 'field.header.stamp',
                       'field.header.frame_id', 'field.sentence', 'rand', 'North', 'N-label', 'West', 'W-label',
                       'rand_1', 'rand_2', 'rand_3', 'rand_4', 'rand_5', 'rand_6', 'rand_7', 'rand_8', 'rand_9']
        gps['time'] = pd.to_datetime(gps['field.header.stamp'] / 1000000000, unit='s')
        gps['file_name'] = gps_filename

        gyro_filename = "./Data/gyro_" + str(i) + ".csv"
        gyro = getData(gyro_filename)
        gyro['time'] = pd.to_datetime(gyro['field.header.stamp'] / 1000000000, unit='s')
        gyro['file_name'] = gyro_filename
        gyro['type'] = "gyr"

        acc_data = acc[['time', 'field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
                        'field.angular_velocity.z',
                        'field.linear_acceleration.x', 'field.linear_acceleration.y',
                        'field.linear_acceleration.z', 'file_name', 'type']]
        gyro_data = gyro[
            ['time', 'field.header.stamp', 'field.angular_velocity.x', 'field.angular_velocity.y',
             'field.angular_velocity.z',
             'field.linear_acceleration.x', 'field.linear_acceleration.y',
             'field.linear_acceleration.z', 'file_name', 'type']]

        gps_data = gps[['time', 'field.header.stamp', 'North', 'West', 'file_name']]

        acc_path_data = acc_path_data.append(acc_data, ignore_index=True)
        gps_path_data = gps_path_data.append(gps_data, ignore_index=True)
        gyro_path_data = gyro_path_data.append(gyro_data, ignore_index=True)
        # imu_path_data = imu_path_data.append(acc_data, ignore_index=True)
        # imu_path_data = imu_path_data.append(gyro_data, ignore_index=True)

    # initial postions
    # initial_position = np.matrix(0,1,)

    acc_path_data['delta_time'] = acc_path_data['field.header.stamp'].diff() / 1000000000
    acc_path_data.loc[0, 'delta_time'] = 0
    acc_path_data['delta_x'] = acc_path_data['field.linear_acceleration.x'] * acc_path_data['delta_time'] * acc_path_data['delta_time']
    acc_path_data['delta_z'] = acc_path_data['field.linear_acceleration.z'] * acc_path_data['delta_time'] * acc_path_data['delta_time']

    gyro_path_data['delta_time'] = gyro_path_data['field.header.stamp'].diff() / 1000000000
    gyro_path_data.loc[0, 'delta_time'] = 0
    gyro_path_data['delta_angle'] = gyro_path_data['field.angular_velocity.y'] * gyro_path_data['delta_time']
    # gyro_path_data['delta_angle'] = np.deg2rad(gyro_path_data['delta_angle'])
    # print(gyro_path_data['delta_angle'])

    imu_path_data = pd.concat([acc_path_data, gyro_path_data], axis=0, ignore_index=True).sort_values('field.header'
                                                                                                      '.stamp',
                                                                                                      ignore_index=True)
    x = []
    y = []
    x.append(0)
    y.append(0)
    heading = 0
    j = 1
    for i in range(1, len(imu_path_data)):
        if imu_path_data.loc[i, 'type'] == 'gyr':
            if math.isnan(imu_path_data.loc[i, 'delta_angle']):
                imu_path_data.loc[i, 'delta_angle'] = 0
            heading = heading + imu_path_data.loc[i, 'delta_angle']
        else:
            if math.isnan(imu_path_data.loc[i, 'delta_x']) or math.isnan(imu_path_data.loc[i, 'delta_z']):
                imu_path_data.loc[i, 'delta_x'] = 0
                imu_path_data.loc[i, 'delta_z'] = 0
            x.append(x[j - 1] + (imu_path_data.loc[i, 'delta_x'] * np.cos(np.pi/2 - heading)))
            y.append(y[j - 1] + (imu_path_data.loc[i, 'delta_z'] * np.sin(np.pi/2 - heading)))
            # if (imu_path_data.loc[i, 'delta_x'] * np.cos(np.pi - heading)) < 0:
            #     print(j)
            j += 1

    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig('imu.png')










    # acc_path_data_sample = acc_path_data[acc_path_data['field.header.stamp'] < 1627578040895700000]
    # imu_path_data = pd.concat([acc_path_data, gyro_path_data], axis=0, ignore_index=True).sort_values(
    #     'field.header.stamp', ignore_index=True)
    # # imu_numpy = imu_path_data.to_numpy()
    # imu_path_data_sample = imu_path_data[imu_path_data['field.header.stamp'] < 1627578040895700000]
    # # print(imu_path_data_sample)
    # imu_path_data_sample['x'] =''
    # imu_path_data_sample['z'] = ''
    # imu_path_data_sample.loc[0, 'x'] = 0
    # imu_path_data_sample.loc[0, 'z'] = 0
    # for i in range(1, len(imu_path_data_sample)):
    #     if imu_path_data_sample.loc[i, 'type'] == 'gyr':
    #         if math.isnan(imu_path_data_sample.loc[i - 1, 'delta_x']):
    #             imu_path_data_sample.loc[i, 'delta_x'] = 0
    #             imu_path_data_sample.loc[i, 'delta_z'] = 0
    #         else:
    #             imu_path_data_sample.loc[i, 'delta_x'] = imu_path_data_sample.loc[i - 1, 'delta_x'] * np.cos(
    #                 imu_path_data_sample.loc[i, 'delta_angle']) + imu_path_data_sample.loc[i - 1, 'delta_z'] * np.sin(
    #                 imu_path_data_sample.loc[i, 'delta_angle'])
    #             imu_path_data_sample.loc[i, 'delta_z'] = -imu_path_data_sample.loc[i - 1, 'delta_x'] * np.sin(
    #                 imu_path_data_sample.loc[i, 'delta_angle']) + imu_path_data_sample.loc[i - 1, 'delta_z'] * np.cos(
    #                 imu_path_data_sample.loc[i, 'delta_angle'])
    #         imu_path_data_sample.loc[i, 'x'] = imu_path_data_sample.loc[i - 1, 'x']
    #         imu_path_data_sample.loc[i, 'z'] = imu_path_data_sample.loc[i - 1, 'z']
    #     else:
    #         imu_path_data_sample.loc[i, 'x'] = imu_path_data_sample.loc[i - 1, 'x'] + imu_path_data_sample.loc[
    #             i - 1, 'delta_x']
    #         imu_path_data_sample.loc[i, 'z'] = imu_path_data_sample.loc[i - 1, 'z'] + imu_path_data_sample.loc[
    #             i - 1, 'delta_z']
    #
    #
    # imu_path_data_sample.to_csv('imu_numpy.csv')
    # # imu_path_data_sample['x'] = imu_path_data_sample['delta_x'].cumsum()
    # # imu_path_data_sample['z'] = imu_path_data_sample['delta_z'].cumsum()
    # # gps_path_data['delta_x'] = gps_path_data['North'].diff()
    # # gps_path_data['delta_y'] = gps_path_data['West'].diff()
    # # gps_path_data['x'] = gps_path_data['delta_x'].cumsum()
    # # gps_path_data['y'] = gps_path_data['delta_y'].cumsum()
    # gps_path_data_sample = gps_path_data[gps_path_data['field.header.stamp'] < 1627578040895700000]
    # gps_path_data_sample['lat'] = np.deg2rad(gps_path_data_sample['North'])
    # # print(gps_path_data['lat'])
    # gps_path_data_sample['lon'] = np.deg2rad(gps_path_data_sample['West'])
    # gps_path_data_sample['lon'] = gps_path_data_sample['lon'] - 180
    # # print(gps_path_data['lon'])
    # # R = 6371  # radius of the earth
    # # gps_path_data['x_utm'] = R * np.cos(gps_path_data['lat']) * np.cos(gps_path_data['lon'])
    # # gps_path_data['y_utm'] = R * np.cos(gps_path_data['lat']) * np.sin(gps_path_data['lon'])
    #
    # gps_path_data_sample['utm_data_x'] = ''
    # gps_path_data_sample['utm_data_y'] = ''
    # for i in range(len(gps_path_data_sample)):
    #     utm_data = utm.from_latlon(gps_path_data_sample.loc[i, 'lat'], gps_path_data_sample.loc[i, 'lon'])
    #     gps_path_data_sample.loc[i, 'utm_data_x'] = utm_data[0]
    #     gps_path_data_sample.loc[i, 'utm_data_y'] = utm_data[1]
    #
    # # gps_path_data['utm'] = gps_path_data.apply(lambda row: utm.from_latlon(gps_path_data['North'], gps_path_data['West']), axis=1)
    # # utm_cols = ['easting', 'northing', 'zone_number', 'zone_letter']
    # # for n, col in enumerate(utm_cols):
    # #     utm_cols[col] = utm_cols['utm'].apply(lambda location: location[n])
    # # df = utm_cols.drop('utm', axis=1)
    #
    # # imu_path_data.sort_values('field.header.stamp')
    # # imu_path_data['delta_time'] = imu_path_data['field.header.stamp'].diff()
    # # imu_path_data['delta_x'] = imu_path_data['field.linear_acceleration.x'] * imu_path_data['delta_time'] * \
    # #                            imu_path_data['delta_time']
    # # imu_path_data['delta_z'] = imu_path_data['field.linear_acceleration.z'] * imu_path_data['delta_time'] * \
    # #                            imu_path_data['delta_time']
    # # imu_path_data['delta_angle'] = imu_path_data['field.angular_velocity.y'] * imu_path_data['delta_time']
    # #
    # # imu_path_data.to_csv('imu_path_data.csv')
    # # for i in range(len(imu_path_data)):
    # #     if imu_path_data.iloc[i, 'type'] == 'gyr':
    # #         if isinstance(imu_path_data.iloc[i-1, 'x'], numbers.Number):
    # #             imu_path_data.iloc[i, 'x'] = imu_path_data.iloc[i-1, 'x'] * np.cos(imu_path_data.iloc[i, 'delta_angle']) + imu_path_data.iloc[i-1, 'z'] * np.sin(imu_path_data.iloc[i, 'delta_angle'])
    # #             imu_path_data.iloc[i, 'z'] = -imu_path_data.iloc[i - 1, 'x'] * np.sin(imu_path_data.iloc[i, 'delta_angle']) + imu_path_data.iloc[i - 1, 'z'] * np.cos(imu_path_data.iloc[i, 'delta_angle'])
    # #         else:
    # #             imu_path_data.iloc[i, 'x'] = 0
    # #             imu_path_data.iloc[i, 'z'] = 0
    # #     elif imu_path_data.iloc[i, 'type'] == 'acc':
    # #         if isinstance(imu_path_data.iloc[i - 1, 'x'], numbers.Number):
    # #             imu_path_data.iloc[i, 'x'] = imu_path_data.iloc[i - 1, 'x'] + imu_path_data.iloc[i, 'delta_x']
    # #             imu_path_data.iloc[i, 'z'] = imu_path_data.iloc[i - 1, 'z'] + imu_path_data.iloc[i, 'delta_z']
    # #         else:
    # #             imu_path_data.iloc[i, 'x'] = 0
    # #             imu_path_data.iloc[i, 'z'] = 0
    #
    # # for i in range(len(imu_path_data)):
    # #     if imu_path_data.iloc[i, 'delta_x'] == 0:
    # #         if imu_path_data[i - 1, 'x']:
    # #             imu_path_data['x'] = imu_path_data[i - 1, 'x']
    # #             imu_path_data['z'] = imu_path_data[i - 1, 'z']
    # #         else:
    # #             imu_path_data['x'] = 0
    # #             imu_path_data['z'] = 0
    # #     else:
    # #         imu_path_data['x'] = imu_path_data['delta_x'].cumsum()
    # #         imu_path_data['z'] = imu_path_data['delta_z'].cumsum()

    # plt.figure(1)
    # plt.plot(acc_path_data_sample['x'], acc_path_data_sample['z'])
    # plt.savefig('imu.png')
    # plt.figure(2)
    # plt.plot(gps_path_data_sample['utm_data_x'], gps_path_data_sample['utm_data_y'])
    # plt.savefig('gps.png')
    plt.show()
