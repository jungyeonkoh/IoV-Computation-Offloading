import numpy as np
import pandas as pd

VEH_COMP_RESOURCE = 500 #(MHz)
VEH_TRAN_POWER = 0.1 #(W)
VEC_COMP_RESOURCE = 6300 #(MHz)
VEC_POWER = 0.007 #(W)
BANDWIDTH = 5 #(MHz)
#KAPPA(fixed)

class Vehicle:
    def __init__(self, id, pos_x, pos_y, velocity):
        self.id = id
        self.pos = [pos_x, pos_y]
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
        self.vehicle_data.set_index("Timestamp", inplace=True)
        self.update_vehicles()

        # .csv파일에서 task 불러오기
        self.task_data = pd.read_csv(load_task_position)
        self.task_data.set_index("Timestamp", inplace=True)
        self.update_tasks()

        # .csv파일에서 server 불러오기
        self.server_data = pd.read_csv(load_server_position)
        for d in self.server_data.values:
            self.servers.append(Server(id=d[0], pos_x=d[1], pos_y=d[2]))

    def update_vehicles(self):
        sub_data = self.vehicle_data.loc[self.update]
        sub_list = sub_data.values
        for d in sub_list:
            if self.update == 1:
                self.vehicles.append(Vehicle(id=d[0], pos_x=d[1], pos_y=d[2], velocity=d[3]))
            else:
                for v in self.vehicles:
                    if d[0] != v.id:
                        continue
                    else:
                        v.pos = [d[1], d[2]]
                        v.v = d[3]
        self.update += 1

    def update_task(self):
        sub_data = self.task_data.loc[self.update]
        sub_list = sub_data.values
        self.tasks = []
        for d in sub_list:
            self.tasks.append(Task(vehicle=d[0], threshold=d[1], input=d[2], comp=d[3], e_weight=d[4]))


    def get_num_vehicle(self):
        return self.num_vehicle


