import environment
import torch
from torch.distributions import Categorical
import numpy as np
from network import Actor
import time
import os


def test(config,rank, global_Actor,counter,max_step):
    print("Start test")
    torch.manual_seed(config["seed"]+rank)
    rewards=[]
    env = environment.Env(**config["EnvironmentParams"], train=False)

    start_time = time.time()
    name=time.strftime('%Y-%m-%d-%Hh-%Mm-%Ss', time.localtime(time.time()))
    f=open("./"+name+".txt",'w')
    f.write(name+"에 시작한 훈련의 log입니다.\n")

    while True:
        score = 0
        step = 0
        state=env.reset()
        hx=torch.zeros(config["EnvironmentParams"]["nv"],config["ActorParams"]["hidden_dim"])
        cx=torch.zeros(config["EnvironmentParams"]["nv"],config["ActorParams"]["hidden_dim"])

        while step < max_step:
            action_prob,partial = global_Actor((torch.FloatTensor(state),(hx,cx)))
            action_dist = Categorical(action_prob)
            action = action_dist.sample()

            next_state, reward = env.step(action,partial)
            state = next_state
            score += np.mean(reward)
            step += 1
            hx=hx.detach()
            cx=cx.detach()
        rewards.append(score)
        f.write("Time: {}, num steps: {}, rewards: {}\n".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value,score/max_step))
        print("test.py: Time: {}, num steps: {}, rewards: {}".format(
            time.strftime("%Hh %Mm %Ss",
                            time.gmtime(time.time() - start_time)),
            counter.value,score/max_step))
        
        if(counter.value>=1000000):
            np.save("reward"+str(env.num_vehicle)+"_test.npy", rewards)