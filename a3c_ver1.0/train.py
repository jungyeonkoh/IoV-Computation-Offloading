import environment
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from network import Actor, Critic
import os

def train(config, rank, counter,lock, batch_size, discount_rate, global_Actor, global_Critic,max_step):
    torch.manual_seed(config["seed"]+rank)
    print("Start "+str(rank)+"process train")
    env = environment.Env(**config["EnvironmentParams"], train=True)

    local_Actor = Actor(**config["ActorParams"])
    local_Critic = Critic(**config["CriticParams"])
    local_Actor.load_state_dict(global_Actor.state_dict())
    local_Critic.load_state_dict(global_Critic.state_dict())
    actor_optimizer = optim.Adam(global_Actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(global_Critic.parameters(), lr=1e-4)

    batch = []
    rewards = []

    hx=torch.zeros(config["EnvironmentParams"]["nv"],config["ActorParams"]["hidden_dim"])
    cx=torch.zeros(config["EnvironmentParams"]["nv"],config["ActorParams"]["hidden_dim"])
    epi=-1
    while True:
        epi+=1
        env.update = 1
        state = env.reset()

        score = 0
        step = 0
        
        while step < max_step:
            # Get action
            
            action_prob,partial = local_Actor((torch.FloatTensor(state),(hx,cx))) # shape: (V, S)
            action_dist = Categorical(action_prob)
            action = action_dist.sample() # server index : 0~
            next_state, reward = env.step(action,partial)
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
                with lock:
                    counter.value+=1

                batch = []
                hx=torch.zeros(config["EnvironmentParams"]["nv"],config["ActorParams"]["hidden_dim"])
                cx=torch.zeros(config["EnvironmentParams"]["nv"],config["ActorParams"]["hidden_dim"])
            else:
                hx=hx.detach()
                cx=cx.detach()

            state = next_state
            score += np.mean(reward)
            step += 1
            #if (step % 1000 == 0 and rank==1):
             #   print("Episode: ", epi, " Step: ", step, " Reward: ", score/step)

        
        #print("Save reward value: ", score/max_step)
        rewards.append(score/max_step)

        # print weight values
        if ((epi % 5) == 4 and rank==1):
            np.save("reward"+str(env.num_vehicle)+"_"+str(epi)+".npy", rewards)
        # save model weights
        if ((epi % 10) == 0 and rank==1):
            save_dir = "./a3c_v"+str(env.num_vehicle)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
 #           torch.save(local_Actor.state_dict(), os.path.join(save_dir, str(epi)+"_local_actor.pt"))
  #          torch.save(local_Critic.state_dict(), os.path.join(save_dir, str(epi)+"_local_critic.pt"))
            torch.save(global_Actor.state_dict(), os.path.join(save_dir, str(epi)+"_global_actor.pt"))
            torch.save(global_Critic.state_dict(), os.path.join(save_dir, str(epi)+"_global_critic.pt"))
        if counter.value>=1000000:
            break
