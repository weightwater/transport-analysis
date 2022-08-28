import pandas as pd
import re
import time
import json
from urllib.request import urlopen
from pprint import pprint
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets


AK = ['pzV79ysREglWVuD25Ill8ky4GrBazGk9']
dir_list = os.listdir('./second_user_data/')


def prepare_startAndend0(time_, lat, lon):
    len_one_num = 0
    start_res = []
    end_res = []
    t, a, o = 0, 0, 0
    for j in range(len(lat)):
        if len(lat[j]) != 1:
            len_one_num += 1
            t = sum(time_[j])/len(time_[j])
            a = sum(lat[j])/len(lat[j])
            o = sum(lon[j])/len(lon[j])

    if len_one_num > 1:
        time_ = [sum(item) / len(item) for item in time_ if len(item) != 1]
        lat = [sum(item) / len(item) for item in lat if len(item) != 1]
        lon = [sum(item) / len(item) for item in lon if len(item) != 1]
        for index in range(len(time_)-1):
            start_res.append([lon[index], lat[index], time_[index]])
            end_res.append([lon[index+1], lat[index+1], time_[index+1]])
    elif len_one_num == 1:
        start_res.append([lon[0][0], lat[0][0], time_[0][0]])
        start_res.append([o, a, t])

        end_res.append([o, a, t])
        end_res.append([lon[-1][0], lat[-1][0], time_[-1][0]])
    else:
        for index in range(len(time_)-1):
            start_unit = []
            end_unit = []

            start_res.append([lon[index], lat[index], time_[index]])
            end_res.append([lon[index + 1], lat[index + 1], time_[index + 1]])

    return start_res, end_res


def distance(a, b):
    return np.sqrt(((b-a)**2).sum())


def resident_analysis0(client_data, range_radius=0.0072, stop_time=400):
    time = np.array(client_data['timestamp'])
    lon = np.array(client_data['longitude'])
    lat = np.array(client_data['latitude'])
    index = 0
    res_time, res_lon, res_lat = [], [], []
    while index < len(time)-1:
        ctime = [time[index]]
        clon = [lon[index]]
        clat = [lat[index]]
        i = 0
        for i in range(index+1, len(time)):
            is_stop = True
            for j in range(index, i):
                if distance(np.array([lat[i], lon[i]]), np.array([lat[j], lon[j]])) > range_radius:
                    is_stop = False
                    break
            if is_stop:
                ctime.append(time[i])
                clon.append(lon[i])
                clat.append(lat[i])
            else:
                break
        res_time.append(ctime[:])
        res_lon.append(clon[:])
        res_lat.append(clat[:])
        index = i
    if index == len(time)-1:
        res_time.append([time[-1]])
        res_lat.append([lat[-1]])
        res_lon.append([lon[-1]])

    # plot distance resident result
    """
        plt.figure()
        for i in range(len(res_time)):
            plt.scatter(res_lat[i], res_lon[i])
        plt.show()
    """

    tmp = []
    for i in range(len(res_time)):
        if res_time[i][-1] - res_time[i][0] < stop_time:
            tmp.append(i)
    tmp_time = []
    tmp_lon = []
    tmp_lat = []
    for i in range(len(res_time)):
        if i in tmp:
            tmp_time.extend([[item] for item in res_time[i]])
            tmp_lon.extend([[item] for item in res_lon[i]])
            tmp_lat.extend([[item] for item in res_lat[i]])
        else:
            tmp_time.append(res_time[i])
            tmp_lon.append(res_lon[i])
            tmp_lat.append(res_lat[i])
    return tmp_time, tmp_lon, tmp_lat


def load_user(name):
    data = pd.read_csv('./second_user_data/'+name, usecols=['timestamp', 'longitude', 'latitude'])
    return data


def resident_analysis(client_data, range_radius=0.0072, stop_time=1800):
    """
    用户驻留分析
    :param client_data: 用户当日信令数据
    :param range_radius: 驻留半径
    :param stop_time: 驻留时间
    :return: 通过驻留分析将信令数据分组
    """
    Seq = []
    stop = True
    unit = []
    index = 0
    start = 0
    end = 0
    # print(len(client_data)-1)
    while index != len(client_data)-1:
        if stop:
            if unit == []:
                unit.append(client_data.loc[index, :])
                start = index
                if index+1 < len(client_data)-1:
                    index += 1
            for item in unit:
                latitude_distance = (client_data.loc[index, 'latitude'] - item['latitude'])**2
                longitude_distance = (client_data.loc[index, 'longitude'] - item['longitude'])**2
                distance = np.sqrt(latitude_distance + longitude_distance)
                time_gap = client_data.loc[index, 'timestamp'] - item['timestamp']
                if not (distance < range_radius and time_gap < stop_time):
                    stop = False
                    index -= 1
                    break
            else:
                unit.append(client_data.loc[index, :])
        else:
            stop = True
            index -= 1
            Seq.append(unit[:])
            unit = []
        index += 1
        # print(index)
    return Seq


def prepare_startAndend(data):
    """
    通过驻留分析之后的数据获取用户在今天的移动的终点和始点的列表
    :param data: 驻留分析之后的数据
    :return: 起点和终点的列表
    """
    point = []
    unit = [[], [], []]
    for group in data:
        for item in group:
            # print(item['timestamp'], item['longitude'], item['latitude'])
            unit[0].append(item['timestamp'])
            unit[1].append(item['longitude'])
            unit[2].append(item['latitude'])
        point.append(unit[:])
        unit = [[], [], []]

    start, end = [], []
    for i in range(len(point)-1):
        start_nums = len(point[i][0])
        start_longitude = sum(point[i][1]) / start_nums
        start_latitude = sum(point[i][2]) / start_nums

        end_nums = len(point[i+1][0])
        end_longtitude = sum(point[i+1][1]) / end_nums
        end_latitude = sum(point[i+1][2]) / end_nums

        start.append([start_longitude, start_latitude])
        end.append([end_longtitude, end_latitude])

    return start, end


def url_test():
    url_drive = 'http://api.map.baidu.com/direction/v2/transit?origin=40.056878,116.30815&destination=31.222965,121.505821&ak='+AK[0]
    res = json.loads(urlopen(url_drive).read())
    pprint(res['result']['routes'][0])


def recommend_road_bus(start, end):
    """
    通过调用百度的路径规划api获取这条路线的公交出行路径
    :param start: 起始位置的经纬度列表
    :param end: 终点位置的经纬度列表
    :return: 路线的经纬度列表以及对应路线所消耗时间的列表
    """
    url_drive = 'http://api.map.baidu.com/direction/v2/transit?origin=' \
                + str(start[0]) + ',' + str(start[1]) \
                + '&destination=' + str(end[0]) + ',' + str(end[1]) \
                + '&ak=' + AK[0]
    res = json.loads(urlopen(url_drive).read())

    if len(res['result']['routes']) == 0:
        return []

    road_data = res['result']['routes']
    roads = []
    times = []
    # pprint(road_data)
    for road in road_data:
        path = []
        lat = []
        lng = []
        for step in road['steps']:
            path.append(list(map(float, re.split(r'[;|,]', step[0]['path']))))
        path = sum(path, [])
        lat = path[0::2]
        lng = path[1::2]
        roads.append([lat, lng])
        times.append(road['duration'])

    # ----------just plot ------------ #
    """
    plt.figure()

    # print(start[0], end[0])
    # print(roads[0][0], roads[0][1])
    plt.scatter([start[1], end[1]], [start[0], end[0]], marker='*', s=100)
    plt.scatter(roads[0][0], roads[0][1], alpha=0.5)

    # plt.show()
    """
    # -------------------------------- #

    return roads, times


def recommend_road_ride(start, end):
    url_ride = 'http://api.map.baidu.com/direction/v2/riding?' \
               'origin=' + str(start[0]) + ',' + str(start[1]) + \
               '&destination=' + str(end[0]) + ',' + str(end[1]) + \
               '&ak=' + AK[0]
    res = json.loads(urlopen(url_ride).read())

    try:
        road_data = res['result']['routes']
    except:
        return [], []
    roads = []
    times = []
    for road in road_data:
        path = []
        lat = []
        lng = []
        for step in road['steps']:
            path.append(list(map(float, re.split(r'[;|,]', step['path']))))
        path = sum(path, [])
        lat = path[0::2]
        lng = path[1::2]
        roads.append([lat, lng])
        times.append(road['duration'])

        # ----------just plot ------------ #
        """
        plt.figure()

        # print(start[0], end[0])
        # print(roads[0][0], roads[0][1])
        plt.scatter([start[1], end[1]], [start[0], end[0]], marker='*', s=100)
        plt.scatter(roads[0][0], roads[0][1], alpha=0.5)

        plt.show()
        """
        # -------------------------------- #

    return roads, times


def recommend_road_car(start, end):
    url_car = 'http://api.map.baidu.com/direction/v2/driving?' \
               'origin=' + str(start[0]) + ',' + str(start[1]) + \
               '&destination=' + str(end[0]) + ',' + str(end[1]) + \
               '&ak=' + AK[0]

    for i in range(10):
        try:
            res = json.loads(urlopen(url_car).read())
        except ConnectionResetError:
            time.sleep(0.5)
        else:
            time.sleep(0.1)
            break

    road_data = res['result']['routes']
    # pprint(road_data)
    roads = []
    times = []
    for road in road_data:
        path = []
        lat = []
        lng = []
        for step in road['steps']:
            path.append(list(map(float, re.split(r'[;|,]', step['path']))))
        path = sum(path, [])
        lat = path[0::2]
        lng = path[1::2]
        roads.append([lat, lng])
        times.append(road['duration'])

        # ----------just plot ------------ #
        """
        plt.figure()

        print(start[0], end[0])
        print(roads[0][0], roads[0][1])
        plt.scatter([start[1], end[1]], [start[0], end[0]], marker='*', s=100)
        plt.scatter(roads[0][0], roads[0][1], alpha=0.5)

        plt.show()
        """
        # -------------------------------- #
    return roads, times


def analysis_unit(file_name):
    # print(file_name)
    user_data = load_user(file_name)
    t, lot, lai = user_data['timestamp'], user_data['longitude'], user_data['latitude']

    time_, lon, lat = resident_analysis0(user_data)
    start, end = prepare_startAndend0(time_, lon, lat)

    road_data = []
    for i in range(len(start)):
        road = []
        unit = {}
        for j in range(len(t)):
            if t[j] > start[i][-1] and t[j] < end[i][-1]:
                road.append([lot[j], lai[j]])
        unit['routes'] = road
        road_data.append(unit)

    rtm_car, car_roads = time_relate_car(start, end)
    rtm_bus, bus_roads = time_relate_bus(start, end)
    rtm_ride, ride_roads = time_relate_ride(start, end)
    # print(rtm_car, rtm_bus, rtm_ride, sep='\n')

    roads = [car_roads[:], ride_roads[:], bus_roads[:]]
    # pprint(roads)
    res = []

    for i in range(len(start)):
        rtm_all = [max(rtm_car[i]), max(rtm_ride[i]), max(rtm_bus[i])]
        max_index = rtm_all.index(max(rtm_all))
        type_index = [rtm_car[i].index(max(rtm_car[i])), rtm_ride[i].index(max(rtm_ride[i])),
                      rtm_bus[i].index(max(rtm_bus[i]))]

        res.append({'type': max_index, 'start': start[i], 'end': end[i],
                    'routes': roads[max_index][i][type_index[max_index]]})

    return res
    """
    plt.figure()
    plt.scatter([item[0] for item in road_data[0]['routes']], [item[1] for item in road_data[0]['routes']])
    plt.show()
    """

    # bus_roads, bus_time = recommend_road_bus(start[0], end[0])
    # ride_roads, ride_time = recommend_road_ride(start[0], end[0])
    # car_roads, car_time = recommend_road_car(start[0], end[0])


def time_relate_car(start, end):
    rtm = []
    roads = []
    for i in range(len(start)):
        bus_roads, bus_time = recommend_road_car(start[i], end[i])

        roads.append(bus_roads[:])

        rtm_i = []
        i = 0
        signal = end[i][-1] - start[i][-1]
        for time in bus_time:
            if signal > time/2:
                rtm_i.append(1-(abs(signal-time)/signal))
            else:
                rtm_i.append(0)
        rtm.append(rtm_i)
    return rtm, roads


def time_relate_bus(start, end):
    rtm = []
    roads = []
    for i in range(len(start)):
        try:
            bus_roads, bus_time = recommend_road_bus(start[i], end[i])
            roads.append(bus_roads[:])
        except:
            rtm.append([0])
            roads.append([[0], [0]])
        else:
            rtm_i = []
            i = 0
            signal = end[i][-1] - start[i][-1]
            for time in bus_time:
                if signal > time/2:
                    rtm_i.append(1-(abs(signal-time)/signal))
                else:
                    rtm_i.append(0)
            rtm.append(rtm_i)
    return rtm, roads


def time_relate_ride(start, end):
    rtm = []
    roads = []
    for i in range(len(start)):
        ride_roads, ride_time = recommend_road_ride(start[i], end[i])
        if len(ride_time) == 0:
            rtm.append([0])
            roads.append([0])
            continue
        roads.append(ride_roads[:])
        rtm_i = []
        i = 0
        signal = end[i][-1] - start[i][-1]
        for time in ride_time:
            if signal > time/2:
                rtm_i.append(1-(abs(signal-time)/signal))
            else:
                rtm_i.append(0)
        rtm.append(rtm_i)
    return rtm, roads


def my_dbscan(D, eps, MinPts):
    labels = [0]*len(D)
    c = 0
    for p in range(len(D)):

        if labels[p] != 0:
            continue

        neighbors = find_neighbors(D, p, eps)

        if len(neighbors) < MinPts:
            labels[p] = -1
        else:
            c += 1
            labels[p] = c

            for i, n in enumerate(neighbors):
                Pn = neighbors[i]

                if labels[Pn] == 0:
                    labels[Pn] = c

                    pn_neightbors = find_neighbors(D, Pn, eps)
                    if len(pn_neightbors) >= MinPts:
                        neighbors += pn_neightbors
                elif labels[Pn] == -1:
                    labels[Pn] = c

    return labels


def find_neighbors(D, P, eps):
    neighbors = []
    for point in range(len(D)):
        if e_distance(D[P], D[point]) < eps:
            neighbors.append(point)
    return neighbors


def e_distance(a, b):
    x = (a[0]-b[0])**2
    y = (a[1]-b[1])**2
    return np.sqrt(x+y)


def test_dbscan():
    noisy_moons, _  = datasets.make_moons(n_samples=100, noise=0.05, random_state=10)
    dbscan_c = my_dbscan(noisy_moons, eps=0.5, MinPts=5)
    dbscan_c = np.array(dbscan_c)
    print(len(dbscan_c))
    plt.figure()
    plt.scatter(noisy_moons[:, 0], noisy_moons[:, 1], c=dbscan_c, cmap='bwr')
    plt.show()


def get_static_data(num):
    return analysis_unit(dir_list[num])


# test_dbscan()
# print(get_static_data(1))

# user_data = load_user(dir_list[0])
# resident_analysis0(user_data)