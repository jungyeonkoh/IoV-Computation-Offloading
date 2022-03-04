from collections import deque
import numpy as np
import os, pickle

# for debug
def print_experience(obs, action, reward, slot):
    print(obs.tolist())
    print("actions: " + str(action) + "reward: " + str(reward) + " slot: " + slot)

class Memory():
    """
    Used for DRQN algorithm to store the observations of vehicles in the buffer.
    """
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, step_size):
        """
        :param batch_size:
        :param step_size: need to be subsequently chosen
        :return:
        """
        idx = np.random.choice(np.arange(len(self.buffer)-step_size), size=batch_size, replace=False)
        ret = []

        for i in idx:
            temp_ret = []
            for j in range(step_size):
                temp_ret.append(self.buffer[i+j])
            ret.append(temp_ret)
        return ret

    def save(self, experiment_name, slot):
        dir_name = "buffer/" + experiment_name
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        ex_file = open(dir_name + "/" + str(slot), "wb")
        pickle.dump(self.buffer, ex_file)
        ex_file.close()

    def load(self, experiment_name, slot):
        dir_name = "buffer/" + experiment_name
        if not os.path.isdir(dir_name):
            return
        im_file = open(dir_name + "/" + str(slot), "rb+")
        self.buffer = pickle.load(im_file)
        im_file.close()


