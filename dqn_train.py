from temp_env import *
import tensorflow as tf
from collections import deque
import random

num_vehicles = 5
num_servers = 12
sum_len = 100  # length of weighted reward sum
gamma = 0.99  # decay rate of weighted reward sum
num_episodes = 10
num_timesteps = 1000  # number of timesteps in one episode

mem_capacity = 10000
exp_mems = [deque([], maxlen=mem_capacity) for i in range(num_vehicles)]
batch_size = 64


alloc_unit = 0.1  # the proportion of the task that the vehicle processes (that is allocated to the vehicle)
                  # is in the interval [0, 1].
                  # we consider only integer multiples of alloc_unit in this interval [0, 1]
input_size = 2 * num_servers + 2  # length of state vector
num_possible_allocs = 1 / alloc_unit + 1  # number of possible allocation proportions
output_size = num_possible_allocs * num_servers # number of possible actions


dqns = []
for i in range(num_vehicles):
    dqn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_size)
    ])
    dqns.append(dqn)


def choose_action(dqn, state):

    action = dqn(state)
    return action  # to do: epsilon, argmax, reshape


def get_weighted_sum(index, reward_record):
    weighted_sum = 0
    for i in range(sum_len):
        weighted_sum += reward_record[index + i] * gamma ** i
    return weighted_sum


def timestep(step_count, reward_record, exps):
    rews = []
    states = env.construct_state()
    actions = []

    # choose action for each vehicle
    for v_index in range(num_vehicles):
        action = choose_action(dqns[v_index], states[v_index])
        actions.append(action)

    for v_index in range(num_vehicles):
        rews.append(env.calculate_reward(v_index, actions[v_index]))
    env.update_vehicle()
    env.update_task()
    next_states = env.construct_state()
    rews_sum = sum(rews)
    reward_record.append(rews_sum)

    # store experience in memory
    for v_index in range(num_vehicles):
        exp = [states[v_index], actions[v_index], rews_sum, next_states[v_index]]
        # exp_mems[v_index].push(exp)
        exps[v_index].append(exp)

    if len(reward_record) >= sum_len:
        index = len(reward_record) - sum_len
        w_sum = get_weighted_sum(index, reward_record)
        for v_index in range(num_vehicles):
            exp_mems[v_index].append(exps[index] + [w_sum])
    # experience is in the format:
    # [state, action, reward, next state, weighted reward sum]

    # if step_count >= batch_size + sum_len:
    # to do: if deque is long enough, train parameters in dqns



step_count = 1

for i in range(num_episodes):
    reward_record = []
    exps = [[] for i in range(num_vehicles)]
    env = Env(num_vehicles, num_servers, 'train.csv', 'simulated_tasks.csv')  # initialize env (set timestep as 1)
    print(f'Starting episode {i}')
    for j in range(num_timesteps):
        timestep(step_count, reward_record, exps)
        step_count += 1



