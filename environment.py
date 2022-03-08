import numpy as np
import pandas as pd

VEH_COMP_RESOURCE = 500 #(MHz)
VEH_TRAN_POWER = 0.1 #(W)
VEC_COMP_RESOURCE = 6300 #(MHz)
VEC_POWER = 0.007 #(W)
BANDWIDTH = 5 #(MHz)
#KAPPA(fixed)

class Vehicle:
    def __init__(self, id, distance, velocity):
        self.id = id
        self.distance = distance
        self.v = velocity
        self.comp = np.random.normal(VEH_COMP_RESOURCE, 30)
        self.tran = np.random.normal(VEH_TRAN_POWER, 0.01)

class Task:
    def __init__(self, vehicle, threshold, input, comp, e_weight):
        self.vehicle = vehicle
        self.threshold = threshold
        self.input = input
        self.comp = comp
        self.e_weight = e_weight

class Server:
    def __init__(self, id, pos_x, pos_y):
        self.id = id
        self.pos = [pos_x, pos_y]
        self.comp = np.random.normal(VEC_COMP_RESOURCE, 70)
        self.power = np.random.normal(VEC_POWER, 0.002)
        self.crowd = 0 #init as 0 (N_j)

class Env:
    def __init__(self, nv, ns, load_vehicle_position, load_task_position, load_server_position):
        self.num_vehicle = nv
        self.vehicles = []
        self.num_server = ns
        self.servers = []
        self.tasks = []

        self.update = 1

        # .csv파일에서 vehicle 불러오기
        self.vehicle_data = pd.read_csv(load_vehicle_position)
        self.vehicle_data.set_index("TIMESTAMP", inplace=True)
        self.update_vehicle()

        # .csv파일에서 task 불러오기
        self.task_data = pd.read_csv(load_task_position)
        self.task_data.set_index("Timestamp", inplace=True)
        self.update_task()

        # .csv파일에서 server 불러오기
        self.server_data = pd.read_csv(load_server_position)
        for d in self.server_data.values:
            self.servers.append(Server(id=d[0], pos_x=d[1], pos_y=d[2]))

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
        self.update += 1

    def update_task(self):
        sub_data = self.task_data.loc[self.update]
        sub_list = sub_data.values
        self.tasks = []
        for d in sub_list:
            self.tasks.append(Task(vehicle=d[0], threshold=d[1], input=d[2], comp=d[3], e_weight=d[4]))

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
            state_vector_by_vehicle = []

            server_vector = []
            for s in range(self.num_server):
                server_vector.append(self.servers[s].crowd)

            state_vector_by_vehicle.append(self.tasks[v].threshold)
            state_vector_by_vehicle.append(self.vehicles[v].velocity)
            state_vector_by_vehicle.append(self.tasks[v].input)
            state_vector_by_vehicle.append(self.tasks[v].comp)
            state_vector_by_vehicle.append(self.vehicles[v].distance)
            state_vector_by_vehicle.append(server_vector)
            state_vector.append(state_vector_by_vehicle)
        return state_vector

    def calculate_reward(self, vehicle, action): # 논문 수정하
        """
        Calculates the reward based on the action of the vehicle.
        """
        reward = self.tasks[vehicle].threshold #TODO
        return reward

    def train_step(self):
        """
        Step function of the environment.
        Calculates the rewards based on the action taken by the vehicles.
        :return:
            rewards: concatenated reward of each vehicle for the taken actions
        """
        rews = np.zeros(self.num_vehicle)

        self.update_vehicle()
        self.update_task()
        state = self.construct_state()
        action = np.zeros((self.num_vehicle, 3)) # 논문 수정하기: action = [float, float, int] (vehicle, server, #server)
        for v in range(self.num_vehicles):
            #action[v] = model.infer_action() #TODO
            rews[v] = self.calculate_reward(v, action[v])
        return rews



