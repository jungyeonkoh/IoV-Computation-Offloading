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

seed = 1

def train(config, env, rank, episode_size, step_size, batch_size, discount_rate, global_Actor, global_Critic):
    torch.manual_seed(seed+rank)

    local_Actor = Actor(**config["ActorParams"])
    local_Critic = Critic(**config["CriticParams"])
    local_Actor.load_state_dict(global_Actor.state_dict())
    local_Critic.load_state_dict(global_Critic.state_dict())
    actor_optimizer = optim.Adam(global_Actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(global_Critic.parameters(), lr=1e-4)

    batch = []
    rewards = []


    for epi in range(episode_size):
        env.update = 1
        state = env.construct_state()

        score = 0
        step = 0
        while step < step_size:
            # Get action
            action_prob = local_Actor(torch.FloatTensor(state)) # shape: (V, S)
            action_dist = Categorical(action_prob)
            action = action_dist.sample() # server index : 0~
            next_state, reward = env.step(action)
            done = np.zeros_like(reward) if len(batch) == batch_size - 1 else np.ones_like(reward)
            action_prob_temp = []
            for i in range(len(action)):
                action_prob_temp.append(action_prob[i][action[i]])
                reward[i] /= 100

            batch.append([state, next_state, reward, action_prob_temp, done])

            if len(batch) >= batch_size:
                state_buffer = []
                next_state_buffer = []
                reward_buffer = []
                action_prob_buffer = []
                done_buffer = []

                for item in batch:
                    state_buffer.append(item[0])
                    next_state_buffer.append(item[1])
                    reward_buffer.append(item[2])
                    action_prob_buffer.append(item[3])
                    done_buffer.append(item[4])

                state_buffer = torch.FloatTensor(state_buffer) # (batch_size, V, state_size)
                next_state_buffer = torch.FloatTensor(next_state_buffer)
                reward_buffer = torch.FloatTensor(reward_buffer).unsqueeze(-1) # (batch_size, V, 1)
                done_buffer = torch.FloatTensor(done_buffer).unsqueeze(-1) # (batch_size, V, 1)

                value_state = local_Critic(state_buffer).squeeze(1) # (batch_size, V, 1)
                value_next_state = local_Critic(next_state_buffer).squeeze(1) # (batch_size, V, 1)
                Q = reward_buffer + discount_rate * value_next_state * done_buffer
                A = Q - value_state

                # update Critic
                critic_optimizer.zero_grad()
                critic_loss = F.mse_loss(value_state, Q.detach()) # constant
                critic_loss.backward(retain_graph=True)
                for global_param, local_param in zip(global_Critic.parameters(), local_Critic.parameters()):
                    global_param._grad = local_param.grad
                critic_optimizer.step()

                # update Actor
                actor_optimizer.zero_grad()
                actor_loss = 0
                for idx, prob in enumerate(action_prob_buffer):
                    for i in range(len(prob)):
                        actor_loss += -A[idx][i] * torch.log(prob[i])
                actor_loss /= len(action_prob_buffer)
                actor_loss.backward()

                for global_param, local_param in zip(global_Actor.parameters(), local_Actor.parameters()):
                    global_param._grad = local_param.grad
                actor_optimizer.step()

                local_Actor.load_state_dict(global_Actor.state_dict())
                local_Critic.load_state_dict(global_Critic.state_dict())
                batch = []

            state = next_state
            score += np.mean(reward)
            step += 1
            if (step % 1000 == 0):
                print("Episode: ", epi, " Step: ", step, " Reward: ", score/step)

        print("Save reward value: ", score/step_size)
        rewards.append(score/step_size)

        # print weight values
        if epi % 5 == 4:
            np.save("reward"+str(env.num_vehicle)+"_"+str(epi)+".npy", rewards)
        # save model weights
        if epi % 10 == 0:
            print("Save model")
            save_dir = "./a3c_v"+str(env.num_vehicle)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            torch.save(local_Actor.state_dict(), os.path.join(save_dir, str(epi)+"_local_actor.pt"))
            torch.save(local_Critic.state_dict(), os.path.join(save_dir, str(epi)+"_local_critic.pt"))
            torch.save(global_Actor.state_dict(), os.path.join(save_dir, str(epi)+"_global_actor.pt"))
            torch.save(global_Critic.state_dict(), os.path.join(save_dir, str(epi)+"_global_critic.pt"))
    return rewards

def test(env, step_size, print_reward_interval, global_Actor):
    iteration = 1
    while True:
        score = 0
        step = 0
        env.update = 1
        state = env.construct_state()

        while step < step_size:
            action_prob = global_Actor(torch.FloatTensor(state))
            action_dist = Categorical(action_prob)
            action = action_dist.sample()

            next_state, reward = env.step(action)
            state = next_state
            score += np.mean(reward)
            step += 1

            if step % print_reward_interval == 0:
                print("Iteration: ", iteration, " Step: ", step, " Reward: ", score/step)
        iteration += 1

def nn(env, step_size):
    step = 0
    score = 0

    while step < step_size:
        rewards = []
        actions = []

        for i in range(env.num_server):
            env.servers[i].crowd = 1
        for i in range(env.num_vehicle):
            action = np.argmin(env.vehicles[i].distance)
            actions.append(action)
            env.servers[action].crowd += 1
        for i in range(env.num_vehicle):
            reward = env.calculate_reward(i, actions[i], 0.)
            rewards.append(reward/100)
        env.update_vehicle()
        env.update_task()
        score += np.mean(rewards)
        step += 1
        if step % 1000 == 0:
            print(step, " : ", score / step)

def rand(env, step_size):
    step = 0
    score = 0

    while step < step_size:
        rewards = []
        actions = []

        for i in range(env.num_server):
            env.servers[i].crowd = 1
        for i in range(env.num_vehicle):
            action = np.random.randint(0, env.num_server)
            actions.append(action)
            env.servers[action].crowd += 1
        for i in range(env.num_vehicle):
            reward = env.calculate_reward(i, actions[i], 0.)
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
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    config = yaml.load(open("./experiment.yaml"), Loader=yaml.FullLoader)

    env = environment.Env(**config["EnvironmentParams"], train=True)
    # test_env = environment.Env(**config["EnvironmentParams"], train=False)
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    global_Actor = Actor(**config["ActorParams"])
    global_Critic = Critic(**config["CriticParams"])
    global_Actor.share_memory()
    global_Critic.share_memory()

    isTrain = config.setdefault("isTrain", True)
    experiment_name = config.setdefault("experiment_name", "")
    episode_size = config.setdefault("episode_size", 1000)
    step_size = config.setdefault("step_size", 10000)
    batch_size = config.setdefault("batch_size", 128)
    discount_rate = config.setdefault("discount_rate", 0.99)
    print_reward_interval = config.setdefault("print_reward_interval", 1000)

    #nn(env, step_size)
    #rand(env, step_size)

    print("==========")
    print("Experiment: " + experiment_name)
    processes = []
    process_num = 1
    mp.set_start_method("spawn")
    print("MP start method: ", mp.get_start_method())
    print("==========")

    # p = mp.Process(target=test, args=(env, step_size, print_reward_interval, global_Actor))
    # p.start()
    # processes.append(p)
    for rank in range(process_num):
        p = mp.Process(target=train, args=(config, env, rank, episode_size, step_size, batch_size, discount_rate, global_Actor, global_Critic))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

