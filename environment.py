import numpy as np
import pandas as pd
import math

# Parameter
# Distance = ~10000m
#
VEH_COMP_RESOURCE = 50 #(MHz)
VEH_TRAN_POWER = 1000 #scaling #0.1(W)
VEC_COMP_RESOURCE = 6300 #(MHz)
VEC_POWER = 0.007 #(W)
BANDWIDTH = 5 #(MHz)
PATH_FADE = 1.75 #scaling #3.75
KAPPA = 10 ** -6 #원래 10^-11~10^-27 (결과에 따라 scaling할것)

class Vehicle:
    def __init__(self, id, distance, velocity):
        self.id = id
        self.distance = distance
        self.v = velocity
        self.comp = np.random.normal(VEH_COMP_RESOURCE, 3)
        self.tran = np.random.normal(VEH_TRAN_POWER, 10)

class Task:
    def __init__(self, vehicle, threshold, input, comp, e_weight):
        self.vehicle = vehicle
        self.threshold = threshold
        self.input = input
        self.comp = comp
        self.e_weight = e_weight

class Server:
    def __init__(self, id):
        self.id = id
        self.comp = np.random.normal(VEC_COMP_RESOURCE, 70)
        self.power = np.random.normal(VEC_POWER, 0.002)
        self.crowd = 1 #init as 1 (N_j)

class Env:
    def __init__(self, nv, ns, vehicle, vehicle_test, task, task_test, train):
        self.num_vehicle = nv
        self.vehicles = []
        self.num_server = ns
        self.servers = []
        self.tasks = []

        self.update = 1

        if train:
            self.vehicle_data = pd.read_csv(vehicle)
            self.task_data = pd.read_csv(task)
        else:
            self.vehicle_data = pd.read_csv(vehicle_test)
            self.task_data = pd.read_csv(task_test)

        # .csv파일에서 vehicle 불러오기
        self.vehicle_data.set_index("TIMESTAMP", inplace=True)
        self.update_vehicle()
        # .csv파일에서 task 불러오기
        self.task_data.set_index("Timestamp", inplace=True)
        self.update_task()

        # server 불러오기
        for s in range(self.num_server):
            self.servers.append(Server(id=s+1))

    def update_vehicle(self):
        sub_data = self.vehicle_data.loc[self.update]
        sub_list = sub_data.values
        for d in sub_list:
            if self.update == 1:
                distance_vector = []
                for i in range(self.num_server):
                    distance_vector.append(d[2+i])
                self.vehicles.append(Vehicle(id=d[0], velocity=d[1], distance=distance_vector))
            else:
                for v in self.vehicles:
                    if d[0] != v.id:
                        continue
                    else:
                        distance_vector = []
                        for i in range(self.num_server):
                            distance_vector.append(d[2+i])
                        v.distance = distance_vector
                        v.v = d[1]

    def update_task(self):
        sub_data = self.task_data.loc[self.update]
        sub_list = sub_data.values
        self.tasks = []

        # for single vehicle
        #self.tasks.append(Task(vehicle=sub_list[0], threshold=sub_list[1], input=sub_list[2], comp=sub_list[3], e_weight=sub_list[4]))
        for d in sub_list:
            self.tasks.append(Task(vehicle=d[0], threshold=d[1], input=d[2], comp=d[3], e_weight=d[4]))
        self.update += 1
    def construct_state(self):
        """
        Constructs the state to be exploited by the algorithms.
        Returns state vector as an input to the RL model calculated for each vehicle.
        * Prerequisite: update_vehicle(), update_task()
        """
        state_vector = []

        for v in range(self.num_vehicle):
            # 논문 순서따름: threshold, velocity, x_i, y_i, distance, N_j
            # (논문 수정하기: GPS point --> distance btwn vehicles and servers)
            # (논문 수정하기: 1*26 1-dim. vector)
            state_vector_by_vehicle = []

            local_time, local_energy = self.get_local_computing(v) # vehicle index: 0~
            state_vector_by_vehicle.append(local_time)
            state_vector_by_vehicle.append(local_energy)
            for s in range(self.num_server):
                remote_time, remote_energy = self.get_remote_computing(v, s)
                state_vector_by_vehicle.append(remote_time)
                state_vector_by_vehicle.append(remote_energy)

            state_vector.append(state_vector_by_vehicle)
        return state_vector

    # def get_max_tolerance(self, v, s): # Eq 1,2 # ID starts from 1
    #     stay_time = 2 * self.vehicles[v-1].distance[s-1] / self.vehicles[v-1].v
    #     return min(stay_time, self.tasks[v-1].threshold)

    def get_transmission_rate(self, v, s): # vehicle index: 0~, server index: 0~
        shared_bandwidth = BANDWIDTH / self.servers[s].crowd
        log = self.vehicles[v].tran * ((self.vehicles[v].distance[s] / 1000) ** (-PATH_FADE))
        log /= self.servers[s].crowd
        return shared_bandwidth * math.log2(log+1)

    def get_local_computing(self, v): # vehicle index: 0~
        time = self.tasks[v].comp / self.vehicles[v].comp
        energy = KAPPA * (self.vehicles[v].comp ** 2) * self.tasks[v].comp
        return time, energy

    def get_remote_computing(self, v, s): # vehicle index: 0~ / server index: 0~
        trans = self.tasks[v].input / self.get_transmission_rate(v,s)
        comp = self.tasks[v].comp / (self.servers[s].comp / self.servers[s].crowd)
        time = trans + comp
        energy = self.vehicles[v].tran * (10 ** -4) * trans + self.servers[s].power * comp # ~0.01
        return time, energy

    def calculate_reward(self, vehicle, action, assign_prob): # 논문 수정하기 / 수식 이상함
        """
        Calculates the reward based on the action of the vehicle.
        """
        reward = 15
        local_time, local_energy = self.get_local_computing(vehicle)
        remote_time, remote_energy = self.get_remote_computing(vehicle, action)
        time = (1-self.tasks[vehicle].e_weight) * (assign_prob * local_time + (1-assign_prob) * remote_time)
        energy = self.tasks[vehicle].e_weight * (assign_prob * local_energy + (1-assign_prob) * remote_energy)
        return reward - time - energy

    def step(self, action):
    #def step(self, action, assign_prob): # action(server) index: 0~
        """
        Step function of the environment.
        Calculates the rewards based on the action taken by the vehicles.
        :return:
            next_state
            rewards: concatenated reward of each vehicle for the taken actions
        """
        for i in range(self.num_server):
            self.servers[i].crowd = 1
        for i in range(self.num_vehicle):
            self.servers[action[i]].crowd += 1

        rewards = []
        for i in range(self.num_vehicle):
            reward = self.calculate_reward(i, action[i], 0.)
            #reward = self.calculate_reward(i, action[i], assign_prob[i])
            rewards.append(reward.item())

        self.update_vehicle()
        self.update_task()
        next_state = self.construct_state()
        return next_state, rewards
