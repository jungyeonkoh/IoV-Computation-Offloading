import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import environment as envs
from A3C_model import ActorCritic

def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    env = envs.Env(args.nv,args.ns,args.load_vehicle_position_test,args.load_task_position_test)

    model = ActorCritic(26,args.ns)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state).float()
    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(args.nv, 128)
            hx = torch.zeros(args.nv, 128)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit,assign_rate, (hx, cx) = model((state,(hx, cx)))
        prob = F.softmax(logit, dim=-1)

        action = (prob.multinomial(num_samples=1).detach()+1).numpy()
        assign_rate=assign_rate.numpy()
        action = np.hstack((assign_rate,action)).tolist()
        reward,state=env.train_step(action)
        reward_sum+=np.sum(reward)
        if (episode_length==args.test_step):
            done==True



        if done:
            print("Time {}, num steps {},".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value,reward_sum))
            reward_sum = 0
            episode_length = 0
            state = env.reset()
            state = torch.from_numpy(state).float()
            time.sleep(60)


