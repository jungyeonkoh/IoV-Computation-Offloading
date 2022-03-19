import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import environment as envs
from A3C_model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)
    env = envs.Env(args.nv,args.ns,args.load_vehicle_position,args.load_task_position)

    model = ActorCritic(26,args.ns)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()


    state = env.reset()
    state = torch.from_numpy(state).float()
    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(args.nv, 128)
            hx = torch.zeros(args.nv, 128)
        else:
            cx = cx.detach()
            hx = hx.detach()


        values = []
        log_probs = []
        rewards = []
        entropies = []


        for step in range(args.num_steps):
            sum_reward=0

            episode_length += 1
            value, logit,assign_rate, (hx, cx) = model((state,(hx, cx)))
                                            
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = (prob.multinomial(num_samples=1).detach()+1).numpy()
            assign_rate=assign_rate.detach().numpy()
            action = np.hstack((assign_rate,action)).tolist()
            log_prob = log_prob.gather(1, prob.multinomial(num_samples=1).detach())
            reward,state=env.train_step(action)
            sum_reward=np.sum(reward)
            
            # reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if episode_length==args.train_step:
                episode_length = 0
                state = env.reset()
                done=True

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(sum_reward)

            if done:
                break

        R = torch.zeros(args.nv, 1)
        if not done:
            value, _, _, _ = model((state, (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (torch.mean(policy_loss) + args.value_loss_coef * torch.mean(value_loss)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        done=False
