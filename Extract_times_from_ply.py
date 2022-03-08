import csv
import os
import re
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import geopandas as gpd
import utm

from pyproj import Proj

from natsort import natsorted


def main():
    # # #Extract timestamps that were captured while creating the ply files as csv
    # # ply_path = "/home/ramya/Downloads/ply_only"
    # # ply_files = natsorted(os.listdir(ply_path))
    # #
    # # list_of_times = []
    # #
    # # for f in ply_files:
    # #     file_name = os.path.basename(os.path.join(ply_path, f))
    # #     timestamp = file_name[:-4]
    # #     list_of_times.append(timestamp)
    # #
    # # list_of_times.sort()
    # #
    # # list_of_times_df = pd.DataFrame(list_of_times)
    # # list_of_times_df.columns = ['field.header.stamp']
    # # # list_of_times_df['field.header.stamp'] = list_of_times_df['field.header.stamp'].astype(int)
    # # # list_of_times_df['time'] = pd.to_datetime(list_of_times_df['field.header.stamp'] / 1000000000, unit='s')
    # # list_of_times_df['file_name'] = 'ply'
    # # # list_of_times_df.to_csv('ply_times.csv', index=False)
    # # # # print(list_of_times_df)
    # # #
    # # gps_files_path = "/home/ramya/Downloads/Day_2"
    # #
    # # gps_file_all = pd.DataFrame(columns=['field.header.stamp', 'North', 'West', 'file_name'])
    # #
    # # for i in range(0, 11):
    # #     gps_file_name = "gps_" + str(i) + ".csv"
    # #     gps_file = pd.read_csv(os.path.join(gps_files_path, gps_file_name), skiprows=range(1), header=None)
    # #     gps_file.columns = ['time_stamp', 'field.header.seq', 'field.header.stamp',
    # #                         'field.header.frame_id', 'field.sentence', 'rand', 'North', 'N-label', 'West', 'W-label',
    # #                         'rand_1', 'rand_2', 'rand_3', 'rand_4', 'rand_5', 'rand_6', 'rand_7', 'rand_8', 'rand_9']
    # #
    # #     gps_file['file_name'] = "gps_" + str(i)
    # #     gps_file = gps_file[['field.header.stamp', 'North', 'West', 'file_name']]
    # #
    # #     gps_file_all = gps_file_all.append(gps_file, ignore_index=True)
    # #
    # # gps_file_all['field.header.stamp'].astype(int)
    # # list_of_times_df['field.header.stamp'].astype(int)
    # #
    # # df_for_interpolation = pd.concat([gps_file_all, list_of_times_df], axis=0, ignore_index=True).sort_values(by=['field.header.stamp'])
    # #
    # # df_for_interpolation.to_csv('temp1.csv', index=False)
    # #
    # df_for_interpolation = pd.read_csv('temp.csv')
    # df_for_interpolation['North'] = df_for_interpolation['North'].interpolate()
    # df_for_interpolation['West'] = df_for_interpolation['West'].interpolate()
    # df_for_interpolation.to_csv('temp.csv', index=False)

    #
    # gps_file_all['time'] = pd.to_datetime(gps_file_all['field.header.stamp'] / 1000000000, unit='s')
    # gps_file_all['North_deg'] = gps_file_all['North'] / 100
    # gps_file_all['West_deg'] = gps_file_all['West'] / 100
    #
    # gps_file_all['North_deg'] = gps_file_all['North_deg'].astype(int)
    # gps_file_all['West_deg'] = gps_file_all['West_deg'].astype(int)
    #
    # gps_file_all['North_minute_decimal'] = gps_file_all['North'] % 100
    # gps_file_all['West_minute_decimal'] = gps_file_all['West'] % 100
    #
    # gps_file_all['North_dd'] = gps_file_all['North_deg'] + (gps_file_all['North_minute_decimal'] / 60)
    # gps_file_all['West_dd'] = (-1) * (gps_file_all['West_deg'] + (gps_file_all['West_minute_decimal'] / 60))
    #
    # gps_file_all['lat'] = np.deg2rad(gps_file_all['North_dd'])
    # gps_file_all['lon'] = np.deg2rad(gps_file_all['West_dd'])
    #
    # myProj = Proj("+proj=utm +zone=20U, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    # lon = np.array(gps_file_all['lon'])
    # lat = np.array(gps_file_all['lat'])
    # lon_, lat_ = myProj(lon, lat)
    # # plt.plot(lon_, lat_)
    # # plt.show()
    #
    # gps_file_all['lon_utm'] = lon_
    # gps_file_all['lat_utm'] = lat_
    #
    # gps_file_all = gps_file_all[
    #     ['time', 'field.header.stamp', 'North', 'West', 'North_dd', 'West_dd', 'lat', 'lon', 'lat_utm', 'lon_utm']]
    # gps_file_all['file_name'] = 'gps'
    #
    # gps_file_all.to_csv('gps_all.csv', index=False)
    #
    ply_times = pd.read_csv('ply_times.csv')
    gps_times = pd.read_csv('gps_all.csv')

    df_for_interpolation = pd.concat([ply_times, gps_times], axis=0, ignore_index=True)
    df_for_interpolation = df_for_interpolation.sort_values('field.header.stamp')
    df_for_interpolation.to_csv('temp.csv', index=False)
    df_for_interpolation['North'] = df_for_interpolation['North'].interpolate()
    df_for_interpolation['West'] = df_for_interpolation['West'].interpolate()

    df_for_interpolation['North_deg'] = df_for_interpolation['North'] / 100
    df_for_interpolation['West_deg'] = df_for_interpolation['West'] / 100

    df_for_interpolation['North_deg'] = df_for_interpolation['North_deg'].astype(int)
    df_for_interpolation['West_deg'] = df_for_interpolation['West_deg'].astype(int)

    df_for_interpolation['North_minute_decimal'] = df_for_interpolation['North'] % 100
    df_for_interpolation['West_minute_decimal'] = df_for_interpolation['West'] % 100

    df_for_interpolation['North_dd'] = df_for_interpolation['North_deg'] + (df_for_interpolation['North_minute_decimal'] / 60)
    df_for_interpolation['West_dd'] = (-1) * (df_for_interpolation['West_deg'] + (df_for_interpolation['West_minute_decimal'] / 60))

    df_for_interpolation['lat'] = np.deg2rad(df_for_interpolation['North_dd'])
    df_for_interpolation['lon'] = np.deg2rad(df_for_interpolation['West_dd'])

    myProj = Proj("+proj=utm +zone=20U, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    lon = np.array(df_for_interpolation['lon'])
    lat = np.array(df_for_interpolation['lat'])
    lon_, lat_ = myProj(lon, lat)
    # plt.plot(lon_, lat_)
    # plt.show()

    df_for_interpolation['lon_utm'] = lon_
    df_for_interpolation['lat_utm'] = lat_

    df_for_interpolation.to_csv('temp1.csv')

    ply_times_extract = df_for_interpolation.loc[df_for_interpolation['file_name'] == 'ply']
    ply_times_extract.to_csv('ply_gps.csv')

    plt.plot(df_for_interpolation['lon'], df_for_interpolation['lat'])
    plt.show()


if __name__ == "__main__":
    main()
