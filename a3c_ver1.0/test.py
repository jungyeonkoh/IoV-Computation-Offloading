import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import environment as envs
from Actor import Actor
import time

def test(rank, args, global_Actor, counter):
    torch.manual_seed(args.seed + rank)
    env = envs.Env(args.nv,args.ns,args.load_vehicle_position,args.load_task_position)
    start_time = time.time()
    name=time.strftime('%Y-%m-%d-%Hh-%Mm-%Ss', time.localtime(time.time()))
    f=open("./"+name+".txt",'w')
    f.write(name+"에 시작한 훈련의 log입니다.\n")
    while True:
        step=0
        s=env.reset()
        score=0            

        step=0
        while(step < args.test_step):
            a_prob = global_Actor(torch.from_numpy(s).float())
            a_distrib = Categorical(a_prob)
            a = a_distrib.sample()
            action=(a.detach()+1).numpy()
            partial_rate=np.zeros_like(action)
            actions = np.column_stack([partial_rate,action]).tolist()
            reward,s_prime =env.train_step(actions)
            score+=torch.mean(torch.from_numpy(reward).float())
            s=s_prime
            step+=1
        f.write("Time: {}, num steps: {}, rewards: {}\n".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value,score/args.train_step))
        print("Time: {}, num steps: {}, rewards: {}".format(
            time.strftime("%Hh %Mm %Ss",
                            time.gmtime(time.time() - start_time)),
            counter.value,score/args.train_step))
        state = env.reset()
        state = torch.from_numpy(state).float()
        time.sleep(10)
              
        
