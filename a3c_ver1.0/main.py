import environment
import yaml
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import os
import numpy as np
from network import Actor, Critic
import os
from test import test
from train import train
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def nn(env):
    step = 0
    score = 0

    while step < max_step:
        rewards = []
        for i in range(env.num_vehicle):
            action = np.argmin(env.vehicles[i].distance)
            reward = env.calculate_reward(i, action, 0.)
            rewards.append(reward/100)
        env.update_vehicle()
        env.update_task()
        score += np.mean(rewards)
        step += 1
        if step % 1000 == 0:
            print(step, " : ", score / step)

def rand(env):
    step = 0
    score = 0

    while step < max_step:
        rewards = []
        for i in range(env.num_vehicle):
            action = np.random.randint(0, env.num_server)
            reward = env.calculate_reward(i, action, 0.)
            rewards.append(reward / 100)
        env.update_vehicle()
        env.update_task()
        score += np.mean(rewards)
        step += 1
        if step % 1000 == 0:
            print(step, " : ", score / step)

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

if __name__ == '__main__':


    config = yaml.load(open("./experiment.yaml"), Loader=yaml.FullLoader)
    seed=config["seed"]
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    #env = environment.Env(**config["EnvironmentParams"], train=True)

    #nn(env)
    #rand(env)

    #test_env = environment.Env(**config["EnvironmentParams"], train=False)
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    global_Actor = Actor(**config["ActorParams"])
    global_Critic = Critic(**config["CriticParams"])
    global_Actor.share_memory()
    global_Critic.share_memory()

    isTrain = config.setdefault("isTrain", True)
    experiment_name = config.setdefault("experiment_name", "")
    #episode_size = config.setdefault("episode_size", 1000)
    step_size = config.setdefault("step_size", 10000)
    batch_size = config.setdefault("batch_size", 128)
    discount_rate = config.setdefault("discount_rate", 0.99)
    #print_reward_interval = config.setdefault("print_reward_interval", 1000)

    processes = []
    num_processes=config["num_processes"]

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(config,num_processes, global_Actor,counter,2000))
    p.start()
    processes.append(p)

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(config, rank, counter, lock, batch_size, discount_rate, global_Actor, global_Critic,config["max_step"]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()