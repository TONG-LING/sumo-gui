'''
Created on 12/12/2017

@author: Liza L. Lemos <lllemos@inf.ufrgs.br>
'''
import socket

import pandas as pd
from matplotlib import pyplot as plt

from base_world import Environment
import traci
# 找一个二进制文件，即寻找可执行文件
import sumolib
from xml.dom import minidom
import sys, os
import subprocess
import atexit
from contextlib import contextmanager
import time
from array import array
import numpy as np
import datetime
import math
from replay import ReplayBuffer
import torch
import random
from openpyxl import load_workbook

from filelock import FileLock
import ast
import config


class SUMOTrafficLights(Environment):

    def __init__(self, cfg_file, port=8813, use_gui=False, batch_size=32):

        super(SUMOTrafficLights, self).__init__()

        # 保存端口与GUI设置，便于在 reset_episode 中指定 TraCI 端口
        self._port = port
        self._use_gui = use_gui

        self.total_NS = 0
        self.total_EW = 0
        self.total_queue_NS = 0
        self.total_queue_EW = 0

        self.replay_buffers = {}
        self.batch_size = batch_size
        self.learners = {}
        self.replay_buffers = {}
        self.__create_env(cfg_file, port, use_gui)
        self.already_counted_ids = set()
        self.counted_vehicles = set()
        self.collected_experiences = {}
        for i in range(4):  # 假设有4个交通信号灯
            self.collected_experiences[str(i)] = []

        self.vehicle_tracking = {}  # {vehicle_id: {'begin_time': time, 'route': route}}
        self.completed_trips = []   # 存储完成的行程数据


    '''
    Create the environment as a MDP. The MDP is modeled as follows:
    * for each traffic light:
    * the  is defined as a vector [current phase, elapsed time of current phase, queue length for each phase]
    * for simplicity, the elapsed time is discretized in intervals of 5s
    * and, the queue length is calculated according to the occupation of the link. 
    * The occupation is discretized in 4 intervals (equally distributed)
    * The number of ACTIONS is equal to the number of phases
    * Currentlly, there are only two phases thus the actions are either keep green time at the current phase or 
    * allow green to another phase. As usual, we call these actions 'keep' and 'change'
    * At each junction, REWARD is defined as the difference between the current and the previous average queue length (AQL)
    * at the approaching lanes, i.e., for each traffic light the reward  is defined as $R(s,a,s')= AQL_{s} - AQL_{s'}$.
    * the transitions between states are deterministic
    '''

    def __create_env(self, cfg_file, port, use_gui):

        # check for SUMO's binaries
        if use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        # register SUMO/TraCI parameters
        self.__cfg_file = cfg_file
        self.__net_file = self.__cfg_file[:self.__cfg_file.rfind("/") + 1] + \
                          minidom.parse(self.__cfg_file).getElementsByTagName('net-file')[0].attributes['value'].value

        # read the network file
        self.__net = sumolib.net.readNet(self.__net_file)

        self.__env = {}

        d = [0, 1]
        # d[0] = 'keep'
        # d[1] = 'change'

        # to each state the actions are the same
        # self.__env[state] has 160 possible variations
        # [idPhase, elapsed time, queue NS, queue EW] = [2, 5, 4, 4]
        # 2 * 5 * 4 * 4 = 160
        # idPhase: 2 phases - NS, EW
        # elapsed time: 30s that are discretize in 5 intervals
        # queue: 0 to 100% discretize in 4 intervals
        # Note: to change the number of phases, it is necessary to change the number of states, e.g. 3 phases: [3, 5, 4, 4, 4]
        # it is also necessary to change the method change_trafficlight
        for x in range(0, 160):
            self.__env[x] = d

        # create the set of traffic ligths
        self.__create_trafficlights()

        self.__create_edges()

    def get_info_E2(decid):  # 传入检测器id信息
        quene_lenth = {}  # 定义排队长度
        occ = {}  # 定义占有率
        lane_length = {}  # 定义车道长度
        for dets in decid:  # 遍历检测器
            lane_id = traci.lanearea.getLaneID(dets)  # 通过检测器获取车道id
            lane_length[lane_id] = traci.lane.getLength(lane_id)  # 获取车道长度并记录
            quene_lenth[lane_id] = traci.lanearea.getJamLengthMeters(dets)  # 获取排队长度并记录
            occ[lane_id] = traci.lanearea.getLastStepOccupancy(dets)  # 获取占有率并记录
        quene_lenth_d = pd.DataFrame.from_dict(quene_lenth, orient='index')  # 将排队长度转换为dataframe
        quene_lenth_d.rename(columns={0: "quene_lenth"}, inplace=True)  # 更改数据标签
        occ_d = pd.DataFrame.from_dict(occ, orient='index')
        occ_d.rename(columns={0: "occ"}, inplace=True)
        lane_length_d = pd.DataFrame.from_dict(lane_length, orient='index')
        lane_length_d.rename(columns={0: "length"}, inplace=True)
        data = pd.concat([quene_lenth_d, occ_d, lane_length_d], axis=1)  # 融合获取的数据信息
        return data

    def __create_trafficlights(self):
        # set of all traffic lights in the simulation
        # each element in __trafficlights correspond to another in __learners
        self.__trafficlights = {}

        # process all trafficlights entries
        junctions_parse = minidom.parse(self.__net_file).getElementsByTagName('junction')
        for element in junctions_parse:
            if element.getAttribute('type') == "traffic_light":
                tlID = element.getAttribute('id').encode('utf-8')
                # print((str(tlID))[2:3])
                tlID = (str(tlID))[2:3]
                # print(tlID)
                # create the entry in the dictionary
                self.__trafficlights[tlID] = {
                    'greenTime': 0,
                    'nextGreen': -1,
                    'yellowTime': -1,
                    'redTime': -1,
                    'current_time': 0,
                    'step': 0,
                    'already_counted_ids': set(),
                    'total_NS': 0,
                    'total_EW': 0
                }

    def reset_episode(self):

        super(SUMOTrafficLights, self).reset_episode()

        # initialise TraCI — robustly close any leftover connections
        try:
            # Prefer explicit cleanup over isLoaded(); close all known labels
            try:
                for _label in list(traci.getConnectionIDs()):
                    try:
                        traci.switch(_label)
                        traci.close(False)
                    except Exception:
                        pass
            except Exception:
                # Fallback: attempt to close current/default connection
                try:
                    traci.close(False)
                except Exception:
                    pass
        except Exception:
            pass

        # 增强鲁棒性：为 SUMO 启动与连接增加最多 5 次重试
        last_err = None
        for attempt in range(5):
            try:
                # Use a unique label for this attempt to avoid 'default' collisions
                _label = f"eval_{os.getpid()}_{int(time.time()*1000)}_{attempt}_{random.getrandbits(32):08x}"
                _args = [
                    self._sumo_binary,
                    "-c", self.__cfg_file,
                    "--scale", "0.8",
                    "--no-warnings",
                ]
                try:
                    from config import SUMO_SEED as _SUMO_SEED
                except Exception:
                    _SUMO_SEED = None
                if _SUMO_SEED is not None:
                    _args += ["--seed", str(_SUMO_SEED)]
                traci.start(_args, label=_label)
                # Ensure subsequent traci.* calls operate on this connection
                try:
                    traci.switch(_label)
                except Exception:
                    pass
                last_err = None
                break
            except Exception as e:
                last_err = e
                try:
                    # Best-effort cleanup for this attempt
                    for _label in list(traci.getConnectionIDs()):
                        try:
                            traci.switch(_label)
                            traci.close(False)
                        except Exception:
                            pass
                except:
                    pass
                # 轻微退避，避免端口/进程竞争
                try:
                    import time as _time
                    _time.sleep(1.0 + attempt * 0.5)
                except:
                    pass
        if last_err is not None:
            # 最终仍失败，抛出原始异常
            raise last_err
        # traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

        # reset traffic lights attributes
        for tlID in self.get_trafficlights_ID_list():
            self.__trafficlights[tlID]['greenTime'] = 0
            self.__trafficlights[tlID]['nextGreen'] = -1
            self.__trafficlights[tlID]['yellowTime'] = -1
            self.__trafficlights[tlID]['redTime'] = -1
            self.__trafficlights[tlID]['current_time'] = 0
            self.__trafficlights[tlID]['step'] = 0

            self.__trafficlights[tlID]['already_counted_ids'].clear()
            self.__trafficlights[tlID]['total_NS'] = 0
            self.__trafficlights[tlID]['total_EW'] = 0

    # define the edges/lanes that are controled for each traffic light
    # the function getControlledLanes() from TRACI, returned the names of lanes doubled
    # that's way is listed here
    def __create_edges(self):
        self._edgesNS = {}
        self._edgesEW = {}

        # 路口0
        self._edgesNS[0] = ['0Ni_0', '0Ni_1', '0Si_0', '0Si_1']
        self._edgesEW[0] = ['0Wi_0', '0Wi_1', '0Ei_0', '0Ei_1']

        # 路口1
        self._edgesNS[1] = ['1Ni_0', '1Ni_1', '1Si_0', '1Si_1']
        self._edgesEW[1] = ['1Wi_0', '1Wi_1', '1Ei_0', '1Ei_1']

        # 路口1
        self._edgesNS[1] = ['1Ni_0', '1Ni_1', '1Si_0', '1Si_1']
        self._edgesEW[1] = ['1Wi_0', '1Wi_1', '1Ei_0', '1Ei_1']
        
        # 路口2
        self._edgesNS[2] = ['2Ni_0', '2Ni_1', '2Si_0', '2Si_1']
        self._edgesEW[2] = ['2Wi_0', '2Wi_1', '2Ei_0', '2Ei_1']
        
        # 路口3
        self._edgesNS[3] = ['3Ni_0', '3Ni_1', '3Si_0', '3Si_1']
        self._edgesEW[3] = ['3Wi_0', '3Wi_1', '3Ei_0', '3Ei_1']

        # 目标优化车道（不再区分内/外圈，仅按你指定的8条）
        # 下：2Ei_0，3Wi_1；右：1Si_1，3Ni_0；上：1Wi_0，0Ei_1 ；左：0Si_0，2Ni_1
        # 为复用现有统计函数，映射到 _inner_ring 四组
        self._inner_ring = {
            'bottom': set(['2Ei_0', '3Wi_1']),
            'right': set(['1Si_1', '3Ni_0']),
            'top': set(['1Wi_0', '0Ei_1']),
            'left': set(['0Si_0', '2Ni_1'])
        }

    # calculates the capacity for each queue of each traffic light
    def __init_edges_capacity(self):
        self._edgesNScapacity = {}
        self._edgesEWcapacity = {}

        for tlID in self.get_trafficlights_ID_list():
            # ~ print '----'
            # ~ print 'tlID', tlID
            lengthNS = 0
            lengthWE = 0
            # 获取交通环境中的信息
            for lane in self._edgesNS[int(tlID)]:
                lengthNS += traci.lane.getLength(lane)
            for lane in self._edgesEW[int(tlID)]:
                lengthWE += traci.lane.getLength(lane)
            lengthNS = lengthNS / 7.5  # vehicle length 5m + 2.5m (minGap)
            lengthWE = lengthWE / 7.5
            self._edgesNScapacity[int(tlID)] = lengthNS
            self._edgesEWcapacity[int(tlID)] = lengthWE

    # https://sourceforge.net/p/sumo/mailman/message/35824947/

    def get_trafficlights_ID_list(self):
        # return a list with the traffic lights' IDs
        return self.__trafficlights.keys()

    # commands to be performed upon normal termination
    def __close_connection(self):
        # Close all active TraCI connections to avoid leaks between episodes
        try:
            for _label in list(traci.getConnectionIDs()):
                try:
                    traci.switch(_label)
                    traci.close(False)
                except Exception:
                    pass
        except Exception:
            # Fallback to closing current connection
            try:
                traci.close(False)
            except Exception:
                pass
        sys.stdout.flush()  # clear standard output

    def close(self):
        try:
            for _label in list(traci.getConnectionIDs()):
                try:
                    traci.switch(_label)
                    traci.close(False)
                except Exception:
                    pass
        except Exception:
            try:
                traci.close(False)
            except Exception:
                pass

    def get_state_actions(self, state):
        self.__check_env()
        # print state
        # print self.__env[state]
        return self.__env[state]

    # check whether the environment is ready to run
    def __check_env(self):
        # check whether the environment data structure was defined
        if not self.__env:
            raise Exception("The traffic lights must be set before running!")

        # discretize the queue occupation in 4 classes equally distributed

    def discretize_queue(self, queue):
        q_class = math.ceil((queue) / 25)
        if queue >= 75:
            q_class = 3

        # percentage
        # ~ if queue < 25:
        # ~ q_class = 0 # 0 - 25%
        # ~ if queue >= 25 and queue < 50:
        # ~ q_class = 1 # 25 - 50%
        # ~ if queue >= 50 and queue < 75:
        # ~ q_class = 2 # 50 - 75%
        # ~ if queue >= 75:
        # ~ q_class = 3 # 75 - 100%

        return int(q_class)

    # change the traffic light phase
    # set yellow phase and save the next green
    def change_trafficlight(self, tlID):
        if traci.trafficlight.getPhase(tlID) == 0:  # NS phase
            traci.trafficlight.setPhase(tlID, 1)
            self.__trafficlights[tlID]["nextGreen"] = 0
        elif traci.trafficlight.getPhase(tlID) == 3:  # EW phase
            traci.trafficlight.setPhase(tlID, 4)
            self.__trafficlights[tlID]["nextGreen"] = 0

    # obs: traci.trafficlights.getPhaseDuration(tlID)
    # it is the time defined in .net file, not the current elapsed time
    def update_phaseTime(self, string, tlID):
        self.__trafficlights[tlID][string] += 1

    # for states
    # 返回的是所有车辆的信息
    # def calculate_queue_size(self, tlID):
    #     minSpeed = 2.8  # 10km/h - 2.78m/s
    #     allVehicles = traci.vehicle.getIDList()
    #
    #     for vehID in allVehicles:
    #         traci.vehicle.subscribe(vehID, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])
    #
    #     info_veh = traci.vehicle.getSubscriptionResults(None)
    #
    #     # VAR_LANE_ID = 81
    #     # VAR_SPEED = 64 Returns the speed of the named vehicle within the last step [m/s]; error value: -1001
    #
    #     qNS = []
    #     qEW = []
    #     if len(info_veh) > 0:
    #         for x in info_veh.keys():
    #             if info_veh[x][81] in self._edgesNS[int(tlID)]:
    #                 qNS.append(x)
    #             if info_veh[x][81] in self._edgesEW[int(tlID)]:
    #                 qEW.append(x)
    #                 # print('qew',qEW)
    #
    #     return [qNS, qEW]
    # for the reward

    # 计算每个红绿灯当前车辆排队数量（安全优化版：使用快照数据确保一致性）
    def calculate_stopped_queue_length(self, tlID, vehicle_data_snapshot=None):
        minSpeed = 1
        
        # 优化：使用传入的数据快照，确保同一时间步内数据一致
        if vehicle_data_snapshot is not None:
            info_veh = vehicle_data_snapshot
        else:
            # 回退：如果没有快照，使用当前所有订阅数据（更稳健）
            try:
                info_veh = traci.vehicle.getAllSubscriptionResults()
            except Exception:
                info_veh = {}

        current_queue_NS = 0
        current_queue_EW = 0

        if info_veh is not None and len(info_veh) > 0:
            # 优化：预先转换为集合，提高查找效率
            edges_NS_set = set(self._edgesNS[int(tlID)])
            edges_EW_set = set(self._edgesEW[int(tlID)])
            
            for veh_id, veh_data in info_veh.items():
                if veh_data[64] <= minSpeed:  # 速度检查
                    lane_id = veh_data[81]
                    if lane_id in edges_NS_set:
                        current_queue_NS += 1
                    elif lane_id in edges_EW_set:
                        current_queue_EW += 1

        return current_queue_NS, current_queue_EW

    def calculate_inner_ring_queues(self, vehicle_data_snapshot=None):
        """
        统计最内圈四条路（top/right/bottom/left）上的停驶车辆数。
        优先使用 lane.getLastStepHaltingNumber，避免订阅丢车；失败时回退到订阅快照统计。
        返回 dict: {'top': int, 'right': int, 'bottom': int, 'left': int}
        """
        counts = {k: 0 for k in ['top', 'right', 'bottom', 'left']}
        try:
            # Primary: lane-level halting numbers (no subscription needed)
            for lane_id in self._inner_ring['top']:
                try:
                    counts['top'] += int(traci.lane.getLastStepHaltingNumber(lane_id))
                except Exception:
                    pass
            for lane_id in self._inner_ring['right']:
                try:
                    counts['right'] += int(traci.lane.getLastStepHaltingNumber(lane_id))
                except Exception:
                    pass
            for lane_id in self._inner_ring['bottom']:
                try:
                    counts['bottom'] += int(traci.lane.getLastStepHaltingNumber(lane_id))
                except Exception:
                    pass
            for lane_id in self._inner_ring['left']:
                try:
                    counts['left'] += int(traci.lane.getLastStepHaltingNumber(lane_id))
                except Exception:
                    pass
            return counts
        except Exception:
            # Fallback: count subscribed vehicles with speed <= 1 m/s in the target lanes
            min_speed = 1.0
            try:
                info_veh = vehicle_data_snapshot if vehicle_data_snapshot is not None else traci.vehicle.getAllSubscriptionResults()
            except Exception:
                info_veh = {}
            if not info_veh:
                return counts
            top_set = self._inner_ring['top']
            right_set = self._inner_ring['right']
            bottom_set = self._inner_ring['bottom']
            left_set = self._inner_ring['left']
            for _, veh_data in info_veh.items():
                try:
                    if veh_data[64] <= min_speed:
                        lane_id = veh_data[81]
                        if lane_id in top_set:
                            counts['top'] += 1
                        elif lane_id in right_set:
                            counts['right'] += 1
                        elif lane_id in bottom_set:
                            counts['bottom'] += 1
                        elif lane_id in left_set:
                            counts['left'] += 1
                except Exception:
                    continue
            return counts

    def calculate_inner_ring_crawl_counts(self, vehicle_data_snapshot=None, speed_threshold=1.0):
        """
        统计最内圈四条路（top/right/bottom/left）上“蠕行车辆”（速度<=speed_threshold）的数量。
        仅用于评估鲁棒性（补足 halting 指标对蠕行的不敏感）。
        返回 dict: {'top': int, 'right': int, 'bottom': int, 'left': int}
        """
        try:
            info_veh = (
                vehicle_data_snapshot
                if vehicle_data_snapshot is not None
                else traci.vehicle.getAllSubscriptionResults()
            )
        except Exception:
            info_veh = {}
        counts = {k: 0 for k in ['top', 'right', 'bottom', 'left']}
        if not info_veh:
            return counts
        top_set = self._inner_ring['top']
        right_set = self._inner_ring['right']
        bottom_set = self._inner_ring['bottom']
        left_set = self._inner_ring['left']
        for _, veh_data in info_veh.items():
            try:
                if veh_data[64] <= speed_threshold:
                    lane_id = veh_data[81]
                    if lane_id in top_set:
                        counts['top'] += 1
                    elif lane_id in right_set:
                        counts['right'] += 1
                    elif lane_id in bottom_set:
                        counts['bottom'] += 1
                    elif lane_id in left_set:
                        counts['left'] += 1
            except Exception:
                continue
        return counts

    

    def get_duration(self, tlID):
        duration = self.__trafficlights[tlID]["greenTime"]
        return duration

    # 获取状态中的相位
    def get_idPhase(self, tlID):
        idPhase = traci.trafficlight.getPhase(tlID)
        return idPhase


    # 获取当前时间间隔内通过检测器的车辆
    # 定义交通信号灯与检测器的对应关系
    def control_traffic(self, tlID):
        # 封装每个信号灯对应的检测器ID
        signal_to_detector_map = {
            '0': {
                'N': ['e1_000', 'e1_001'],
                'S': ['e1_004', 'e1_005'],
                'W': ['e1_002', 'e1_003'],
                'E': ['e1_006', 'e1_007']
            },
            '1': {
                'N': ['e1_008', 'e1_009'],
                'S': ['e1_012', 'e1_013'],
                'W': ['e1_010', 'e1_011'],
                'E': ['e1_014', 'e1_015']
            },
            '2': {
                'N': ['e1_024', 'e1_025'],
                'S': ['e1_028', 'e1_029'],
                'W': ['e1_026', 'e1_027'],
                'E': ['e1_030', 'e1_031']
            },
            '3': {
                'N': ['e1_032', 'e1_033'],
                'S': ['e1_036', 'e1_037'],
                'W': ['e1_034', 'e1_035'],
                'E': ['e1_038', 'e1_039']
            }
        }

        detectors = signal_to_detector_map[tlID]

        # 用于存储当前仿真步骤中去重后的车辆ID
        unique_vehicle_ids = {
            'N': set(),
            'S': set(),
            'W': set(),
            'E': set()
        }
        # 获取对应检测器的车辆数据
        for direction, detector_ids in detectors.items():
            for detector_id in detector_ids:
                vehicle_ids = traci.inductionloop.getLastStepVehicleIDs(detector_id)
                for vehicle_id in vehicle_ids:
                    if vehicle_id not in self.__trafficlights[tlID]['already_counted_ids']:  # 每个信号灯维护自己的ID集合
                        unique_vehicle_ids[direction].add(vehicle_id)
                        self.__trafficlights[tlID]['already_counted_ids'].add(vehicle_id)  # 将新检测到的车辆ID加入已统计集合

        # 返回去重后的车辆ID个数
        n = len(unique_vehicle_ids['N'])
        s = len(unique_vehicle_ids['S'])
        NS = n + s
        w = len(unique_vehicle_ids['W'])
        e = len(unique_vehicle_ids['E'])
        WE = w + e

        return NS, WE

    def get_traffic_state(self, tlID, step):
        cycle_duration = 5
        # 每秒执行control_traffic并累加结果
        ns, we = self.control_traffic(tlID)
        self.__trafficlights[tlID]['total_NS'] += ns
        self.__trafficlights[tlID]['total_EW'] += we
        duration = self.get_duration(tlID)
        idPhase = self.get_idPhase(tlID)

        # 每5秒输出累加结果并重置计数器
        if step % cycle_duration == 0:
            # 在重置前保存当前累加值
            total_NS = self.__trafficlights[tlID]['total_NS']
            total_EW = self.__trafficlights[tlID]['total_EW']

            # 重置累计计数器
            self.__trafficlights[tlID]['total_NS'] = 0
            self.__trafficlights[tlID]['total_EW'] = 0

            # 返回当前周期的累计值
            return idPhase, duration, total_NS, total_EW
        else:
            # 如果不是周期结束，返回当前累加值（不重置）
            return self.get_idPhase(tlID), self.get_duration(tlID), self.__trafficlights[tlID]['total_NS'], \
                self.__trafficlights[tlID]['total_EW']

    def __init_replay_buffers(self):
        self.replay_buffers = {}
        for tlID in self.get_trafficlights_ID_list():
            self.replay_buffers[int(tlID)] = ReplayBuffer()

    def save_vehicle_data(self, produced_data, departed_data):
        # 保存 produced_vehicles
        with FileLock(f"{config.DATA_DIR}/produced_vehicles.csv.lock"):
            produced_df = pd.DataFrame(produced_data, columns=['Time', 'Produced Vehicles'])
            if os.path.exists(f'{config.DATA_DIR}/produced_vehicles.csv'):
                existing_df = pd.read_csv(f'{config.DATA_DIR}/produced_vehicles.csv')
                combined_df = pd.concat([existing_df, produced_df], ignore_index=True)
                # CSV文件没有行数限制，移除行数限制
            else:
                combined_df = produced_df
            combined_df.to_csv(f'{config.DATA_DIR}/produced_vehicles.csv', index=False)

        # 保存 departed_vehicles
        with FileLock(f"{config.DATA_DIR}/departed_vehicles.csv.lock"):
            departed_df = pd.DataFrame(departed_data, columns=['Time', 'Departed Vehicles'])
            if os.path.exists(f'{config.DATA_DIR}/departed_vehicles.csv'):
                existing_df = pd.read_csv(f'{config.DATA_DIR}/departed_vehicles.csv')
                combined_df = pd.concat([existing_df, departed_df], ignore_index=True)
                # CSV文件没有行数限制，移除行数限制
            else:
                combined_df = departed_df
            combined_df.to_csv(f'{config.DATA_DIR}/departed_vehicles.csv', index=False)

    def save_green_time_data(self):
        """保存绿灯时间数据到CSV文件"""
        import csv
        import os
        import datetime
        
        # 创建一个字典，按交通灯ID和相位分组统计平均绿灯时间
        phase_stats = {}
        for record in self.green_time_data:
            tlID, phase, duration, _ = record
            key = (tlID, phase)
            if key not in phase_stats:
                phase_stats[key] = []
            phase_stats[key].append(duration)
        
        # 按交通灯ID整理数据
        summary_by_tl = {}
        for (tlID, phase), durations in phase_stats.items():
            avg_duration = sum(durations) / len(durations)
            if tlID not in summary_by_tl:
                summary_by_tl[tlID] = {"ns_duration": 0, "ns_count": 0, "ew_duration": 0, "ew_count": 0}
            
            # 分开存储南北和东西方向数据
            if phase == 0:  # 南北方向
                summary_by_tl[tlID]["ns_duration"] = avg_duration
                summary_by_tl[tlID]["ns_count"] = len(durations)
            elif phase == 3:  # 东西方向
                summary_by_tl[tlID]["ew_duration"] = avg_duration
                summary_by_tl[tlID]["ew_count"] = len(durations)
        
        # 保存详细的绿灯时间记录
        with open(f'{config.DATA_DIR}/green_time.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['交通灯ID', '相位', '持续时间(秒)', '时间戳'])
            for record in self.green_time_data:
                writer.writerow(record)
        
        # 保存平均绿灯时间统计
        file_path = f'{config.DATA_DIR}/green_time_summary.csv'
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # 只有在文件不存在时才写入表头
            if not file_exists:
                writer.writerow(['交通灯ID', '南北平均持续时间(秒)', '南北切换次数', 
                            '东西平均持续时间(秒)', '东西切换次数', '时间戳'])
            
            # 写入每个交通灯的统计数据
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for tlID, stats in summary_by_tl.items():
                writer.writerow([
                    tlID, 
                    stats["ns_duration"], 
                    stats["ns_count"],
                    stats["ew_duration"], 
                    stats["ew_count"],
                    timestamp
                ])

    def calculate_free_flow_time(self, route):
        """计算路径的自由流行程时间"""
        total_time = 0
        
        for edge_id in route:
            try:
                lane_id = edge_id + '_0'
                edge_length = traci.lane.getLength(lane_id)
                max_speed = traci.lane.getMaxSpeed(lane_id)
                
                if max_speed > 0:
                    edge_time = edge_length / max_speed
                    total_time += edge_time
                else:
                    total_time += edge_length / 13.89  # 默认50km/h
            except:
                total_time += 100 / 13.89  # 默认值
                
        return total_time

    def calculate_free_flow_time_for_distance(self, route, distance_m):
        """仅按已行驶距离 distance_m 计算自由流时间（沿 route，逐边限速累积）"""
        remaining = float(distance_m)
        total_time = 0.0
        for edge_id in route:
            lane_id = edge_id + '_0'
            edge_len = traci.lane.getLength(lane_id)
            vmax = traci.lane.getMaxSpeed(lane_id)
            if vmax is None or vmax <= 0:
                vmax = 13.89  # 50km/h
            take = min(remaining, edge_len)
            if take > 0:
                total_time += take / vmax
                remaining -= take
            if remaining <= 0:
                break
        return total_time


    def run_episode(self, max_steps=-1, exp=None, epoch_idx = 0, mode='train', sample_id=None, save_outputs=True):
        import os  # 添加这一行确保在函数内部可以访问os模块
        import datetime  # 也添加datetime模块，因为后面也会用到
        
        global current_time_all
        self.__check_env()
        # 添加绿灯时间记录列表
        if mode == 'eval':
            self.green_time_data = []  # 存储格式: [tlID, phase, duration, timestamp]

        # getTime returns seconds (float); keep max_steps in seconds
        self._has_episode_ended = False
        self._episodes += 1
        self.reset_episode()

        # Always close TraCI even if an exception occurs mid-episode
        _episode_metrics = None
        # Cleanup is handled explicitly at the end of the episode

        # 目标车道四组每步排队计数序列（用于报表）
        inner_queue_samples = {
            'top': [], 'right': [], 'bottom': [], 'left': []
        }
        # 目标车道四组之和的逐步序列（halting/crawl/robust）
        inner_total_per_step = []
        halting_total_per_step = []
        crawl_total_per_step = []
        robust_total_per_step = []
        # 新增：按方向记录每步“robust”(max(halting,crawl))，用于尾段鲁棒统计
        robust_dir_per_step = {'top': [], 'right': [], 'bottom': [], 'left': []}

        self.__init_edges_capacity()  # initialize the queue capacity of each traffic light
        # self.__create_tlogic()

        # ----------------------------------------------------------------------------------

        current_time = 0
        previousNSqueue = [0] * len(self.get_trafficlights_ID_list())
        previousEWqueue = [0] * len(self.get_trafficlights_ID_list())
        currentNSqueue = [0] * len(self.get_trafficlights_ID_list())
        currentEWqueue = [0] * len(self.get_trafficlights_ID_list())
        currentqueNS = 0
        currentqueEW = 0
        currentqueNSlength = 0
        currentqueEWlength = 0
        new_state = [0] * len(self.get_trafficlights_ID_list())
        state = [0] * len(self.get_trafficlights_ID_list())
        previous_state = [[0, 0, 0, 0, 0] for _ in range(len(self.get_trafficlights_ID_list()))]  # 新增：保存上一状态用于奖励计算
        choose = [0] * len(self.get_trafficlights_ID_list())  # flag: if choose an action
        maxGreenTime = 90  # maximum green time, to prevent starvation
        minGreenTime = 10
        interv_action_selection = 5  # interval for action selection

        # reward_data = []
        # CPMData = [] if mode == 'train' else None
        CPMData = {f'state_reward_{i}': [] for i in range(4)} if mode == 'train' else None

        produced_vehicles_data = [] if mode == 'eval' else None  # 用于记录每秒产生的车辆
        departed_vehicles_data = [] if mode == 'eval' else None  # 用于记录每秒离开的车辆

        # 用于记录低于 5 km/h 的车辆数量
        slow_vehicles_data = [] if mode == 'eval' else None

        tracked_vehicles = set()  # 记录已统计的车辆

        update_epsilon = maxGreenTime * 2  # maxGreenTime *2: to assure that the traffic ligth pass at least one time in each phase
        step = 0
        cycle_duration = 5
        data = []
        slow_data = []
        # fit_r = 0

        # 产生的车辆
        pro_data = []
        # 离开的车辆
        leave_data = []
        # 记录每个时间步的离开车辆数据
        leave_data_summary = []
        ns = []
        we = []
        total_leave_data = []
        # 统计四个交通灯的数据
        total_leave_data = 0

        # main loop
        # 在循环外定义一个集合来跟踪在上一时间步骤存在的车辆
        previous_vehicles = set()

        previous_slow_vehicles = set()  # 用于记录上一步骤速度低于5km/h的车辆

        # 创建一个空的 DataFrame 用于保存结果
        result = pd.DataFrame()
        # 初始化数据存储列表
        traffic_light_data = [] if mode == 'eval' else None

        # 添加一个新的字典来存储每个交通信号灯的经验
        self.collected_experiences = {tlID: [] for tlID in self.get_trafficlights_ID_list()}
        # 新增：延后回填奖励r所需的临时容器与每秒的(state+action)序列
        self.temp_experiences = {tlID: [] for tlID in self.get_trafficlights_ID_list()}
        self.state_action_seq = {tlID: [] for tlID in self.get_trafficlights_ID_list()}
        # 新增：当前动作的持久化（用于非决策秒仍能记录 s+a）
        self.current_action = {tlID: 0 for tlID in self.get_trafficlights_ID_list()}
        self.cpm_last_block_start = None
        self.cpm_last_block_rows = 0

        # 全局车辆订阅（优化：避免重复订阅）
        subscribed_vehicles = set()
        
        while ((max_steps > -1 and traci.simulation.getTime() < max_steps) or max_steps <= -1) and (
                traci.simulation.getMinExpectedNumber() > 0 or traci.simulation.getArrivedNumber() > 0):
            actlist = [0] * len(self.get_trafficlights_ID_list())

            learner_state_action = {}
            traci.simulationStep()

            # 获取当前所有车辆的ID
            all_vehicles = traci.vehicle.getIDList()
            
            # 安全优化：确保所有车辆都被订阅，但避免重复订阅
            current_vehicles = set(all_vehicles)
            new_vehicles = current_vehicles - subscribed_vehicles
            
            # 立即订阅新车辆，确保数据可用性
            for veh_id in new_vehicles:
                try:
                    traci.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])
                    subscribed_vehicles.add(veh_id)
                except:
                    pass
            
            # 清理已离开的车辆订阅
            departed_vehicles_set = subscribed_vehicles - current_vehicles
            subscribed_vehicles -= departed_vehicles_set
            
            # 获取当前步骤的车辆数据快照（关键：确保数据一致性）
            try:
                vehicle_data_snapshot = traci.vehicle.getAllSubscriptionResults() or {}
            except Exception:
                vehicle_data_snapshot = {}

            # 只在评估模式下跟踪车辆延误
            if mode == 'eval':
                # 跟踪新生成的车辆
                departed_vehicles = traci.simulation.getDepartedIDList()
                for veh_id in departed_vehicles:
                    if veh_id not in self.vehicle_tracking:
                        try:
                            route = traci.vehicle.getRoute(veh_id)
                            self.vehicle_tracking[veh_id] = {
                                'begin_time': current_time_all,
                                'route': route
                            }
                        except:
                            pass

                # 处理完成行程的车辆
                arrived_vehicles = traci.simulation.getArrivedIDList()
                for veh_id in arrived_vehicles:
                    if veh_id in self.vehicle_tracking:
                        try:
                            actual_travel_time = current_time_all - self.vehicle_tracking[veh_id]['begin_time']
                            route = self.vehicle_tracking[veh_id]['route']
                            free_flow_time = self.calculate_free_flow_time(route)
                            delay = actual_travel_time - free_flow_time

                            # 添加调试信息
                            # print(f"车辆 {veh_id}:")
                            # print(f"  开始时间: {self.vehicle_tracking[veh_id]['begin_time']}")
                            # print(f"  结束时间: {current_time_all}")
                            # print(f"  路径: {route}")
                            # print(f"  实际行程时间: {actual_travel_time:.2f}秒")
                            # print(f"  自由流时间: {free_flow_time:.2f}秒")
                            # print(f"  延误: {delay:.2f}秒")
                            # print("---")
                            
                            self.completed_trips.append({
                                'delay': delay,
                                'actual_time': actual_travel_time,
                                'free_flow_time': free_flow_time,
                                'weight':1.0,
                                'finished':True
                            })
                            
                            del self.vehicle_tracking[veh_id]
                        except:
                            if veh_id in self.vehicle_tracking:
                                del self.vehicle_tracking[veh_id]


            if mode == 'eval':
                # 记录当前时间产生的车辆
                produced_vehicles = []
                for vid in all_vehicles:
                    if vid not in tracked_vehicles and traci.vehicle.getLaneID(vid) != "":
                        produced_vehicles.append(vid)
                        tracked_vehicles.add(vid)  # 标记为已统计

                if current_time >= 10:
                    # 记录当前时间的车辆数量
                    produced_vehicles_count = len(produced_vehicles)
                    # 统计离开的车辆
                    departed_vehicles = previous_vehicles - set(all_vehicles)
                    departed_vehicles_count = len(departed_vehicles)
                    # 记录产生和离开的车辆数量
                    produced_vehicles_data.append((current_time, produced_vehicles_count))
                    departed_vehicles_data.append((current_time, departed_vehicles_count))
                    # 优化：使用快照数据统计速度低于5km/h的车辆，确保数据一致性
                    slow_vehicles = [vid for vid, data in vehicle_data_snapshot.items() if data[64] < (3 / 3.6)] if vehicle_data_snapshot else []
                    # 当前时间的速度低于5km/h的车辆（去重处理）
                    new_slow_vehicles = set(slow_vehicles) - previous_slow_vehicles  # 新进入低速状态的车辆
                    departed_slow_vehicles = previous_slow_vehicles - set(slow_vehicles)  # 离开低速状态的车辆
                    # 更新记录
                    slow_vehicles_count = len(new_slow_vehicles)  # 统计当前进入低速状态的车辆数
                    slow_vehicles_data.append((current_time, slow_vehicles_count))
                    # 更新previous_slow_vehicles为当前时间的低速车辆
                    previous_slow_vehicles = set(slow_vehicles)

                    # 更新 previous_vehicles 为当前时间步骤的车辆
                    previous_vehicles = set(all_vehicles)

            # total_leave_data = 0

            total_slow = 0

            # 优化：缓存时间和相位查询结果
            current_time_all = traci.simulation.getTime()
            tl_phases = {}
            for tlID in self.get_trafficlights_ID_list():
                tl_phases[tlID] = traci.trafficlight.getPhase(tlID)

            for tlID in self.get_trafficlights_ID_list():
                # 使用缓存的结果
                current_phase = tl_phases[tlID]

                if mode == 'eval':
                    # 保存信号灯数据
                    traffic_light_data.append([tlID, current_time_all, current_phase])

                #统计速度低于3.6km/h的车辆（使用快照数据确保一致性）
                queNS, queEW = self.calculate_stopped_queue_length(tlID, vehicle_data_snapshot)
                total_slow += queEW+queNS
                
                # 统计指定目标车道（四组）的排队长度（halting 与 crawl 双口径）
                halting_counts = self.calculate_inner_ring_queues(vehicle_data_snapshot)
                crawl_counts = self.calculate_inner_ring_crawl_counts(vehicle_data_snapshot, speed_threshold=1.0)

                # 保持样本记录沿用 halting 口径，便于与历史可比
                inner_queue_samples['top'].append(halting_counts['top'])
                inner_queue_samples['right'].append(halting_counts['right'])
                inner_queue_samples['bottom'].append(halting_counts['bottom'])
                inner_queue_samples['left'].append(halting_counts['left'])

                h_sum = halting_counts['top'] + halting_counts['right'] + halting_counts['bottom'] + halting_counts['left']
                c_sum = crawl_counts['top'] + crawl_counts['right'] + crawl_counts['bottom'] + crawl_counts['left']
                r_sum = h_sum if h_sum >= c_sum else c_sum

                # 按方向的 robust 计数
                try:
                    r_top = max(int(halting_counts['top']), int(crawl_counts['top']))
                    r_right = max(int(halting_counts['right']), int(crawl_counts['right']))
                    r_bottom = max(int(halting_counts['bottom']), int(crawl_counts['bottom']))
                    r_left = max(int(halting_counts['left']), int(crawl_counts['left']))
                except Exception:
                    r_top = r_right = r_bottom = r_left = 0
                robust_dir_per_step['top'].append(r_top)
                robust_dir_per_step['right'].append(r_right)
                robust_dir_per_step['bottom'].append(r_bottom)
                robust_dir_per_step['left'].append(r_left)

                inner_total_per_step.append(h_sum)
                halting_total_per_step.append(h_sum)
                crawl_total_per_step.append(c_sum)
                robust_total_per_step.append(r_sum)


                if current_phase == 0 or current_phase == 3:
                    current_time = current_time_all  # 使用缓存的时间
                    self.__trafficlights[tlID]['step'] += 1
                    idPhase, duration, total_NS, total_EW = self.get_traffic_state(tlID,
                                                                                   self.__trafficlights[tlID]['step'])
                    # 关键修复：两次调用使用相同快照数据，确保结果一致
                    queNS, queEW = self.calculate_stopped_queue_length(tlID, vehicle_data_snapshot)
                    data.append([current_time, queNS + queEW])
                    # print(f"tlid:{tlID},curentime:{current_time}, step:{self.__trafficlights[tlID]['step']},idPhase:{idPhase},NS:{total_NS}, WE:{total_EW},queNS:{queNS},queEW:{queEW},greentime:{duration}")
                    new_state[int(tlID)] = [idPhase, total_NS, total_EW, queNS, queEW]

                # 新增：记录每秒的（state + action） 供扩散模型使用
                if mode == 'train' and current_time_all > 10:
                    within_t = int(current_time_all) - 11
                    if 0 <= within_t < 1000:
                        sa_vec = new_state[int(tlID)] + [self.current_action[tlID]]
                        self.state_action_seq[tlID].append(sa_vec)


                # 控制是否在当前时间点进行动作选择
                # 控制是否强制切换信号灯相位
                if self.__trafficlights[tlID]["greenTime"] > 9 and \
                        (self.__trafficlights[tlID]["greenTime"] % interv_action_selection) == 0 \
                        and (traci.trafficlight.getPhase(tlID) == 0 or traci.trafficlight.getPhase(tlID) == 3):

                    state[int(tlID)], action = self._learners[tlID].act_last(new_state[int(tlID)], tlID)

                    learner_state_action[tlID] = [state[int(tlID)], action]
                    #同步持久化当前动作
                    self.current_action[tlID] = action

                    # if green time is equal or more than maxGreenTime, change phase
                    if self.__trafficlights[tlID]["greenTime"] >= maxGreenTime:
                        learner_state_action[tlID] = [state[int(tlID)], 1]

                    choose[int(tlID)] = True  # flag: if choose an action

                    # new_state[int(tlID)] = [0, 0, 0, 0]
                else:
                    choose[int(tlID)] = False
                # 将每个时间步的总离开车辆数据保存



            # 奖励生成策略：基于排队长度变化
            # - 第0个epoch（epoch_idx==0）继续在线按启发式计算 r（用于收集初始样本训练扩散）
            # - 从第1个epoch起，不在线获取 r，延后到仿真结束后用扩散生成并回填
            for tlID in self.get_trafficlights_ID_list():
                if mode == 'train' and epoch_idx == 0:
                    # 方案A：直接使用负排队长度作为奖励（与进化优化目标一致）
                    # new_state = [idPhase, total_NS, total_EW, queNS, queEW]
                    current_queue = new_state[int(tlID)][3] + new_state[int(tlID)][4]  # queNS + queEW
                    
                    # 奖励 = -排队长度（排队越多奖励越负，排队越少奖励越高）

                    r = int(-current_queue) 
                else:
                    r = None  # 延后回填

                # print("r1:", r)


                # state reward action一起
                # if mode == 'train' and current_time_all > 10:
                #     state_reward = new_state[int(tlID)]+[action]+[r]
                #     # CPMData.append((tlID, state_reward))
                #     CPMData[f'state_reward_{tlID}'].append(state_reward)
                # 仅在第0个epoch收集 (s,a,r) 到 CPMData，用于训练扩散模型
                if mode == 'train' and current_time_all > 10:
                    rec_action = self.current_action[tlID]
                    r_record = r if r is not None else 0
                    state_reward = new_state[int(tlID)] + [rec_action] + [r_record]
                    CPMData[f'state_reward_{tlID}'].append(state_reward)
                # 其他epoch不在此写入，改为仿真结束后扩散回填r再离线学习

                # 对应交通信号灯绿灯时间+1
                if traci.trafficlight.getPhase(tlID) == 0 or traci.trafficlight.getPhase(tlID) == 3:
                    self.update_phaseTime('greenTime', tlID)
                    # if choose == True: run the action (change, keep)
                    # else: just calculate the queue length (reward will be the average queue length)
                    if choose[int(tlID)]:
                        # B) RUN ACTION
                        if (learner_state_action[tlID][1] if tlID in learner_state_action else self.current_action[tlID]) == 1:  # TODO: more phases
                            if mode == 'eval':
                                # 记录当前绿灯时间
                                current_phase = traci.trafficlight.getPhase(tlID)
                                green_duration = self.__trafficlights[tlID]['greenTime']
                                current_timestamp = traci.simulation.getTime()
                                
                                # 记录数据: [交通灯ID, 相位(0=南北绿灯,3=东西绿灯), 持续时间, 时间戳]
                                self.green_time_data.append([tlID, current_phase, green_duration, current_timestamp])

                            self.__trafficlights[tlID]['greenTime'] = 0
                            self.__trafficlights[tlID]['step'] = 0
                            # this method must set yellow phase and save the next green phase
                            self.change_trafficlight(tlID)
                            actlist[int(tlID)] = 1


                    if self.__trafficlights[tlID]["greenTime"] > (minGreenTime - 1) and \
                            (self.__trafficlights[tlID]["greenTime"] % interv_action_selection) == 0 and \
                            current_time > 10:

                        within_t = int(current_time_all) - 11
                        if 0 <= within_t < 1000:
                            if epoch_idx == 0:
                                # 仍按原逻辑即时反馈（用于CPMData收集，便于对齐）
                                trafficlight_to_proces_feedback = {
                                    tlID: [r, new_state[int(tlID)], state[int(tlID)]]
                                }
                                self.__process_trafficlights_feedback(trafficlight_to_proces_feedback)
                            else:
                                # 从第1个epoch起，仅记录经验，延后回填r
                                self.temp_experiences[tlID].append({
                                    't': within_t,
                                    'state': state[int(tlID)],
                                    'action': (learner_state_action[tlID][1] if tlID in learner_state_action else self.current_action[tlID]),
                                    'next_state': new_state[int(tlID)]
                                })

                        previousNSqueue[int(tlID)] = currentqueNS
                        previousEWqueue[int(tlID)] = currentqueEW
                        currentqueNS = 0
                        currentqueEW = 0
                        
                        # 更新previous_state用于下次奖励计算
                        previous_state[int(tlID)] = new_state[int(tlID)].copy()

            # self.metrics(arq_tl, current_time)
            queue_list = ""
            for tlID in self.get_trafficlights_ID_list():
                queue_len = round((previousNSqueue[int(tlID)] + previousEWqueue[int(tlID)]), 1)
                queue_list = queue_list + str(queue_len) + ","


            # 保存车速小于3.6km/h的车辆
            slow_data.append((current_time_all, total_slow))
            # 使用 pandas 将数据转换为 DataFrame
            df = pd.DataFrame(slow_data, columns=['Time', 'slow'])

        if mode == 'train':
            if epoch_idx == 0 and CPMData:
                # 仅第0个epoch：保存CPMData用于训练扩散
                df = pd.DataFrame(CPMData)
                file_path = config.CPM_DATA_PATH
                lock_path = file_path + ".lock"
                with FileLock(lock_path):
                    if os.path.exists(file_path):
                        existing_df = pd.read_csv(file_path)
                        combined_df = pd.concat([existing_df, df], ignore_index=True)
                        start_idx = len(existing_df)
                    else:
                        combined_df = df
                        start_idx = 0
                    combined_df.to_csv(file_path, index=False)
                self.cpm_last_block_start = start_idx
                self.cpm_last_block_rows = len(df)
            else:
                # 第1个epoch起：保存本回合的 (s,a) 600步序列，供扩散生成r
                sa_out = {}
                for tl in self.get_trafficlights_ID_list():
                    col = f'state_action_{tl}'
                    seq = self.state_action_seq[tl]
                    sa_out[col] = [str(x) for x in seq]
                sa_df = pd.DataFrame(sa_out)
                sa_path = config.SA_SEQ_PATH
                with FileLock(sa_path + ".lock"):
                    sa_df.to_excel(sa_path, index=False, engine="openpyxl")

                # 把本回合CPMData（r=0占位）追加写入，并记录写入范围
                if CPMData:
                    df = pd.DataFrame(CPMData)
                    file_path = config.CPM_DATA_PATH
                    lock_path = file_path + ".lock"
                    with FileLock(lock_path):
                        if os.path.exists(file_path):
                            existing_df = pd.read_csv(file_path)
                            start_idx = len(existing_df)
                            combined_df = pd.concat([existing_df, df], ignore_index=True)
                        else:
                            start_idx = 0
                            combined_df = df
                        combined_df.to_csv(file_path, index=False)
                    self.cpm_last_block_start = start_idx
                    self.cpm_last_block_rows = len(df)

        elif mode == 'eval':
            # 保存车辆数据
            if save_outputs and produced_vehicles_data and departed_vehicles_data:
                self.save_vehicle_data(produced_vehicles_data, departed_vehicles_data)
            
            # 保存低速车辆数据
            # if slow_vehicles_data:
            #     df = pd.DataFrame(slow_vehicles_data, columns=['Time', 'slow'])
            #     df.to_excel('slow_data_0.xlsx', index=False, engine='openpyxl')
            
            # 保存绿灯时间数据
            if save_outputs and self.green_time_data:
                self.save_green_time_data()
            
            now_t = traci.simulation.getTime()
            active_ids = set(traci.vehicle.getIDList())
            for veh_id, info in list(self.vehicle_tracking.items()):
                if veh_id in active_ids:
                    dist_m = traci.vehicle.getDistance(veh_id)
                    actual_time = now_t - info['begin_time']
                    free_flow_partial = self.calculate_free_flow_time_for_distance(info['route'], dist_m)
                    delay = actual_time - free_flow_partial
                    free_flow_total = self.calculate_free_flow_time(info['route'])
                    weight = min(1.0, free_flow_partial / free_flow_total) if free_flow_total > 0 else 0.0
                    self.completed_trips.append({
                        'delay': delay,
                        'actual_time': actual_time,
                        'free_flow_time': free_flow_partial,
                        'weight': weight,
                        'finished': False
                    })

            # 新版：计算并保存最内圈四边之和的统计（主要指标，沿用 halting 口径）
            if inner_total_per_step:
                avg_inner_sum_val = sum(inner_total_per_step) / len(inner_total_per_step)
                max_inner_sum_val = max(inner_total_per_step)
                min_inner_sum_val = min(inner_total_per_step)

                # 保存排队统计摘要（最内圈四边之和）
                summary_data = {
                    'episode': epoch_idx,
                    'avg_inner_sum': avg_inner_sum_val,
                    'max_inner_sum': max_inner_sum_val,
                    'min_inner_sum': min_inner_sum_val,
                    'total_samples': len(inner_total_per_step),
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                if save_outputs:
                    summary_file_path = config.QUEUE_SUMMARY_PATH
                    summary_df = pd.DataFrame([summary_data])
                    
                    with FileLock(summary_file_path + ".lock"):
                        if os.path.exists(summary_file_path):
                            existing_summary = pd.read_excel(summary_file_path)
                            combined_summary = pd.concat([existing_summary, summary_df], ignore_index=True)
                        else:
                            combined_summary = summary_df
                        combined_summary.to_excel(summary_file_path, index=False, engine='openpyxl')
                print(f"population: Episode {epoch_idx}: avg_inner_sum = {avg_inner_sum_val:.2f}, max_inner_sum = {max_inner_sum_val:.0f}")

            # 评估鲁棒指标（基于 robust_total_per_step）
            avg_tail_robust = None
            p95_tail_robust = None
            max_tail_robust = None
            tail_len = 0
            if mode == 'eval' and robust_total_per_step:
                try:
                    warmup = int(getattr(config, 'EVAL_WARMUP_SEC', 0))
                    tail_sec = int(getattr(config, 'EVAL_TAIL_SEC', 0))
                except Exception:
                    warmup, tail_sec = 0, 0
                series = robust_total_per_step[warmup:] if warmup > 0 else list(robust_total_per_step)
                if tail_sec > 0 and len(series) > tail_sec:
                    tail_series = series[-tail_sec:]
                else:
                    tail_series = series
                if tail_series:
                    tail_len = len(tail_series)
                    avg_tail_robust = float(sum(tail_series) / tail_len)
                    try:
                        import numpy as _np
                        p95_tail_robust = float(_np.percentile(_np.array(tail_series, dtype=float), 95))
                    except Exception:
                        # 简易近似：按排序取 95% 位置
                        sorted_ts = sorted(tail_series)
                        idx = max(0, int(0.95 * (len(sorted_ts)-1)))
                        p95_tail_robust = float(sorted_ts[idx])
                    max_tail_robust = float(max(tail_series))

            # 计算最内圈四条路的时间平均排队长度
            avg_inner_top = float(sum(inner_queue_samples['top']) / len(inner_queue_samples['top'])) if inner_queue_samples['top'] else None
            avg_inner_right = float(sum(inner_queue_samples['right']) / len(inner_queue_samples['right'])) if inner_queue_samples['right'] else None
            avg_inner_bottom = float(sum(inner_queue_samples['bottom']) / len(inner_queue_samples['bottom'])) if inner_queue_samples['bottom'] else None
            avg_inner_left = float(sum(inner_queue_samples['left']) / len(inner_queue_samples['left'])) if inner_queue_samples['left'] else None
            inner_list = [avg_inner_top, avg_inner_right, avg_inner_bottom, avg_inner_left]
            inner_sum = float(sum(v for v in inner_list if v is not None)) if any(v is not None for v in inner_list) else None

            # 不再统计外圈，以上四组即为唯一指标
            
            # 清理trip数据
            if self.completed_trips:
                self.completed_trips.clear()
                self.vehicle_tracking.clear()

        self.__close_connection()
        self._has_episode_ended = True

        if mode == 'eval':
            # 在返回前计算方向级尾段鲁棒（若尚未计算）
            try:
                warmup = int(getattr(config, 'EVAL_WARMUP_SEC', 0))
                tail_sec = int(getattr(config, 'EVAL_TAIL_SEC', 0))
            except Exception:
                warmup, tail_sec = 0, 0
            avg_tail_dir = []
            p95_tail_dir = []
            max_tail_dir = []
            for _k in ['top', 'right', 'bottom', 'left']:
                series = robust_dir_per_step[_k][warmup:] if warmup > 0 else list(robust_dir_per_step[_k])
                if tail_sec > 0 and len(series) > tail_sec:
                    tail_series = series[-tail_sec:]
                else:
                    tail_series = series
                if tail_series:
                    try:
                        import numpy as _np
                    except Exception:
                        _np = None
                    avg_v = float(sum(tail_series) / len(tail_series))
                    if _np is not None:
                        try:
                            p95_v = float(_np.percentile(_np.array(tail_series, dtype=float), 95))
                        except Exception:
                            p95_v = float(sorted(tail_series)[max(0, int(0.95 * (len(tail_series)-1)))])
                    else:
                        p95_v = float(sorted(tail_series)[max(0, int(0.95 * (len(tail_series)-1)))])
                    max_v = float(max(tail_series))
                else:
                    avg_v = None
                    p95_v = None
                    max_v = None
                avg_tail_dir.append(avg_v)
                p95_tail_dir.append(p95_v)
                max_tail_dir.append(max_v)

            _episode_metrics = {
                "average_queue_inner_per_road": inner_list,  # [top, right, bottom, left]
                "average_queue_inner_sum": inner_sum,
                "avg_tail_robust": avg_tail_robust,
                "p95_tail_robust": p95_tail_robust,
                "max_tail_robust": max_tail_robust,
                "tail_steps": tail_len,
                # 新增：方向级尾段鲁棒
                "avg_tail_robust_per_road": avg_tail_dir,
                "p95_tail_robust_per_road": p95_tail_dir,
                "max_tail_robust_per_road": max_tail_dir,
                "sample_id": sample_id
            }
            return _episode_metrics

    def __process_trafficlights_feedback(self, traffic_lights):
        # 收集经验而不是立即学习
        for tlID in traffic_lights.keys():
            experience = self._learners[str(tlID)].feedback_last(traffic_lights[tlID][0], traffic_lights[tlID][1],
                                                    traffic_lights[tlID][2])
            if experience:
                self.collected_experiences[str(tlID)].append(experience)

    def metrics(self, arquivo, current_time):
        minSpeed = 2.8  # 10km/h - 2.78m/s

        # using subcriptions
        allVehicles = traci.vehicle.getIDList()
        for vehID in allVehicles:
            traci.vehicle.subscribe(vehID, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])

        lanes = traci.vehicle.getSubscriptionResults(None)

        # VAR_LANE_ID = 81
        # VAR_SPEED = 64 Returns the speed of the named vehicle within the last step [m/s]; error value: -1001
        # VAR_WAITING_TIME = 122 	Returns the waiting time [s]

        cont_veh_per_tl = [0] * len(self.get_trafficlights_ID_list())
        if len(lanes) > 0:
            for x in lanes.keys():
                for tlID in self.get_trafficlights_ID_list():
                    if lanes[x][64] <= minSpeed:
                        if (lanes[x][81] in self._edgesNS[int(tlID)]) or (lanes[x][81] in self._edgesEW[int(tlID)]):
                            cont_veh_per_tl[int(tlID)] += 1

        # save in a file
        # how many vehicles were in queue in each timestep
        average_queue = 0
        for tlID in self.get_trafficlights_ID_list():
            average_queue = average_queue + cont_veh_per_tl[int(tlID)]
        average_queue = average_queue / float(len(self.__trafficlights))
        arquivo.writelines(
            '%d,%s,%.1f,%d\n' % (current_time, str(cont_veh_per_tl)[1:-1], average_queue, len(allVehicles)))

    def run_step(self):
        raise Exception('run_step is not available in %s class' % self)

    def has_episode_ended(self):
        return self._has_episode_ended

    def __calc_reward(self, state, action, new_state):
        raise Exception('__calc_reward is not available in %s class' % self)

    def finalize_experiences_from_generated_rewards(self, epoch_idx):
        """
        读取生成的奖励文件，
        将本回合临时经验按时间步回填 r，并进行DQN的离线学习。
        额外：对齐(s,a,r)，将本回合 CPMData 追加写入 CPMData.csv。
        """
        file_path = config.GENERATED_REWARD_PATH
        if not os.path.exists(file_path):
            print(f"[WARN] 未找到生成奖励文件: {file_path}")
            return

        gen_df = pd.read_excel(file_path)  # 期望列: reward0,reward1,reward2,reward3

        # 先为离线学习回填 (s,a,s',r)；同时准备 CPMData 输出
        # cpm_out = {f"state_reward_{i}": [] for i in range(4)}

        for tlID in self.get_trafficlights_ID_list():
            col_idx = int(tlID)
            col_name = f"reward{col_idx}"
            if col_name not in gen_df.columns:
                print(f"[WARN] 缺少列 {col_name}, 跳过 tlID={tlID}")
                continue

            experiences = []
            for exp in self.temp_experiences[tlID]:
                t = exp['t']
                if 0 <= t < len(gen_df):
                    r = int(gen_df.iloc[t, col_idx])
                    experiences.append((exp['state'], exp['action'], exp['next_state'], r))
            if experiences:
                print(f"交通信号灯 {tlID}: 回填 {len(experiences)} 条经验并开始离线学习...")
                print(f"  学习前重放池大小: {self._learners[tlID].replay.size}")  
                steps = self._learners[tlID].sequential_learn_current_epoch(experiences)
                print(f"  学习后重放池大小: {self._learners[tlID].replay.size}")  
                print(f"交通信号灯 {tlID}: 完成 {steps} 步学习")

        # 在COMData.csv 原位替换占位0
        try:
            cpm_path = config.CPM_DATA_PATH
            if not os.path.exists(cpm_path):
                print(f"[WARN] 未找到CPMData文件: {cpm_path}")
            else:
                df = pd.read_csv(cpm_path)
                start = getattr(self, 'cpm_last_block_start', None)
                rows = getattr(self, 'cpm_last_block_rows', 0)

                # 如果没有记录范围，则回填到文件末尾len(gen_df)行
                if start is None or rows <= 0:
                    rows = min(len(gen_df), len(df))
                    start = len(df) - rows

                end = min(start + rows, len(df))
                for col_idx in range(4):
                    col_name = f'state_reward_{col_idx}'
                    if col_name not in df.columns:
                        continue
                    for t in range(start, end):
                        cell = df.at[t, col_name]
                        if pd.isna(cell):
                            continue
                        try:
                            lst = ast.literal_eval(cell) if isinstance(cell, str) else list(cell)
                            gen_t = t - start
                            if 0 <= gen_t < len(gen_df):
                                lst[-1] = int(gen_df.iloc[gen_t, col_idx])
                                df.at[t, col_name] = str(lst)
                        except Exception as e:
                            # 跳过解析失败的行
                            pass
                with FileLock(cpm_path + ".lock"):
                    df.to_csv(cpm_path, index=False)
        except Exception as e:
            print(f"[WARN] 回填CPMData失败: {e}")

        # 清空临时容器
        self.temp_experiences = {tl: [] for tl in self.get_trafficlights_ID_list()}
        self.state_action_seq = {tl: [] for tl in self.get_trafficlights_ID_list()}
        self.cpm_last_block_start = None
        self.cpm_last_block_rows = 0

    def build_experiences_from_rewards_array(self, episode_index: int, rewards_array):
        """
        用“内存中的生成奖励数组”回填临时经验池，组装为 (s, a, s', r) 元组列表。
        - rewards_array: np.ndarray 或 torch.Tensor，形状 [T=1000, 4]，与你生成的奖励格式一致
        - return: dict, key 是 tlID（或 str(tlID)），value 是该路口的 List[(state, action, next_state, reward)]
        说明：
        - 完全复用 finalize_experiences_from_generated_rewards 的组装规则：
        以临时经验池 temp_experiences[tlID] 中的时间步 t 为索引，从 rewards_array[t, col_idx] 取 r。
        - 不写入 CPMData，不写入 self.replay_buffers；只返回给调用方由其决定如何训练。
        """
        # 1) 统一把 rewards_array 变成 numpy，方便按 [t, col] 索引
        import numpy as np
        if hasattr(rewards_array, "detach"):  # torch.Tensor
            rewards_array = rewards_array.detach().cpu().numpy()
        else:
            rewards_array = np.asarray(rewards_array)

        # 2) 构造 tlID -> 奖励列索引 的映射（与 finalize 保持一致）
        #    如果你在 finalize 里已经有 col_idx 的计算方式，最好抽成同一个函数；这里默认用路口 ID 的顺序枚举（0..3）
        tl_list = self.get_trafficlights_ID_list()
        tl_to_col_idx = {str(tlID): i for i, tlID in enumerate(tl_list)}  # 与 reward_0..3 列对应

        # 3) 组装各路口的经验列表
        results = {}  # tlID(str) -> List[(s, a, s', r)]
        T = rewards_array.shape[0]  # 期望为 1000
        for tlID in tl_list:
            key = str(tlID)
            col_idx = tl_to_col_idx[key]
            exps = []
            if key in self.temp_experiences:
                for exp in self.temp_experiences[key]:
                    t = exp['t']
                    if 0 <= t < T:
                        # 与 finalize 一致：按时间 t 和该路口的 reward 列取值
                        r = int(rewards_array[t, col_idx])
                        exps.append((exp['state'], exp['action'], exp['next_state'], r))
            results[key] = exps

        # 4) 不清空 temp_experiences，不写入回放池；完全由调用方决定如何使用
        return results

    def write_cpm_data_from_sample_result(self, sample_result, epoch, round_idx):
        """
        根据选定样本结果，将CPMData.csv中的0占位符替换为generated_reward.xlsx中的奖励值
        """
        import pandas as pd
        import ast
        from filelock import FileLock
        
        # 读取最佳样本的奖励数据
        file_path = config.GENERATED_REWARD_PATH
        if not os.path.exists(file_path):
            print(f"[WARN] 未找到生成奖励文件: {file_path}")
            return

        gen_df = pd.read_excel(file_path)  # 期望列: reward0,reward1,reward2,reward3

        # 在CPMData.csv 原位替换占位0
        try:
            cpm_path = config.CPM_DATA_PATH
            if not os.path.exists(cpm_path):
                print(f"[WARN] 未找到CPMData文件: {cpm_path}")
            else:
                df = pd.read_csv(cpm_path)
                start = getattr(self, 'cpm_last_block_start', None)
                rows = getattr(self, 'cpm_last_block_rows', 0)

                # 如果没有记录范围，则回填到文件末尾len(gen_df)行
                if start is None or rows <= 0:
                    rows = min(len(gen_df), len(df))
                    start = len(df) - rows

                end = min(start + rows, len(df))
                for col_idx in range(4):
                    col_name = f'state_reward_{col_idx}'
                    if col_name not in df.columns:
                        continue
                    for t in range(start, end):
                        cell = df.at[t, col_name]
                        if pd.isna(cell):
                            continue
                        try:
                            lst = ast.literal_eval(cell) if isinstance(cell, str) else list(cell)
                            gen_t = t - start
                            if 0 <= gen_t < len(gen_df):
                                lst[-1] = int(gen_df.iloc[gen_t, col_idx])
                                df.at[t, col_name] = str(lst)
                        except Exception as e:
                            # 跳过解析失败的行
                            pass
                with FileLock(cpm_path + ".lock"):
                    df.to_csv(cpm_path, index=False)
        except Exception as e:
            print(f"[ERROR] 处理CPMData.csv时出错: {e}")
