import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import environment as envs
import torch.optim as optim
from Actor import Actor
from Critic import Critic

def train(rank, args, global_Actor,global_Critic, counter, lock):
    torch.manual_seed(args.seed + rank)
    env = envs.Env(args.nv,args.ns,args.load_vehicle_position,args.load_task_position)

    local_Actor=Actor(state_space=(args.ns*2)+2,
                    action_space=args.ns,
                    num_hidden_layer=args.hidden_layer_num,
                    hidden_dim=args.hidden_dim_size)
    local_Critic = Critic(state_space=(args.ns*2)+2,
                    num_hidden_layer=args.hidden_layer_num,
                    hidden_dim=args.hidden_dim_size)
    local_Actor.load_state_dict(global_Actor.state_dict())
    local_Critic.load_state_dict(global_Critic.state_dict())


    batch=[]
    a_prob_list=[]

  # Set Optimizer
    actor_optimizer = optim.Adam(global_Actor.parameters(), lr=args.lr)
    critic_optimizer = optim.Adam(global_Critic.parameters(), lr=args.lr)

    while True:
        done=False
        score=0

        step=0
        s=env.reset()
        while(not done) and (step < args.train_step):

          a_prob = local_Actor(torch.from_numpy(s).float())
          a_distrib = Categorical(a_prob)
          
          a = a_distrib.sample()
          action=(a.detach()).numpy()
          partial_rate=np.zeros_like(action)
          actions = np.column_stack([partial_rate,action]).tolist()
          
          reward,s_prime =env.train_step(actions)
          done_mask = np.zeros_like(reward) if len(batch)==(args.batch_size-1) else np.ones_like(reward)

          for i in range(len(a)):
            a_prob_list.append(a_prob[i][a[i]])
            reward[i]/=100 
          batch.append([s,reward,s_prime,a_prob_list,done_mask])
          a_prob_list=[]
          if len(batch)>=args.batch_size:
            with lock:
              counter.value+=1
            s_buf = []
            s_prime_buf = []
            r_buf = []
            prob_buf = []
            done_buf = []

            for item in batch:
              s_buf.append(item[0])
              r_buf.append(item[1])
              s_prime_buf.append(item[2])
              prob_buf.append(item[3])
              done_buf.append(item[4])

            s_buf = torch.FloatTensor(np.array(s_buf))
            r_buf = torch.FloatTensor(np.array(r_buf)).unsqueeze(-1)
            s_prime_buf = torch.FloatTensor(np.array(s_prime_buf))
            done_buf = torch.FloatTensor(np.array(done_buf)).unsqueeze(-1)

            v_s = local_Critic(s_buf)
            v_prime = local_Critic(s_prime_buf)

            Q = r_buf+args.discount_rate*v_prime.detach()*done_buf # value target
            A =  Q - v_s   
            
            # Update Critic
            critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(v_s, Q.detach())
            critic_loss.backward(retain_graph=True)
            for global_param, local_param in zip(global_Critic.parameters(), local_Critic.parameters()):
                global_param._grad = local_param.grad

            critic_optimizer.step()
            # Update Actor
            actor_optimizer.zero_grad()
            actor_loss = 0
            for idx, prob in enumerate(prob_buf):
              for i in range(len(prob)):
                actor_loss += -A[idx][i] * torch.log(prob[i])
            actor_loss /= len(prob_buf) 
            actor_loss.backward()

            for global_param, local_param in zip(global_Actor.parameters(), local_Actor.parameters()):
                global_param._grad = local_param.grad
            actor_optimizer.step()

            local_Actor.load_state_dict(global_Actor.state_dict())
            local_Critic.load_state_dict(global_Critic.state_dict())

            batch = []

          s = s_prime
          score += reward
          step += 1
          if(step==args.train_step):
            break

        
              