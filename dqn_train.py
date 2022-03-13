from temp_env import *
# from environment import *
import tensorflow as tf
from collections import deque
import random
import numpy as np

num_vehicles = 3
num_servers = 12
sum_len = 10  # length of weighted reward sum   100
gamma = 0.8  # decay rate of weighted reward sum
num_episodes = 10
num_timesteps = 1000  # number of timesteps in one episode

mem_capacity = 500
exp_mems = [deque([], maxlen=mem_capacity) for i in range(num_vehicles)]
batch_size = 64

alloc_unit = 0.1  # the proportion of the task that the vehicle processes (that is allocated to the vehicle)
# is in the interval [0, 1].
# we consider only integer multiples of alloc_unit in this interval [0, 1]
input_size = 2 * num_servers + 2  # length of state vector
num_possible_allocs = 1 / alloc_unit + 1  # number of possible allocation proportions
output_size = num_possible_allocs * num_servers  # number of possible actions

dqns = []
for i in range(num_vehicles):
    dqn = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(output_size)
    ])
    dqns.append(dqn)

random_chance = 0.9  # probability of choosing action randomly


def choose_action(dqn, state):
    if random.random() > random_chance * 0.8 ** (step_count / 100):
        qualities = dqn(state)
        action = tf.math.argmax(qualities[0])
        action = int(action)
    else:
        action = random.randrange(0, output_size)
    return action


def get_weighted_sum(index, reward_record):
    weighted_sum = 0
    for i in range(sum_len):
        weighted_sum += reward_record[index + i] * gamma ** i
    return weighted_sum


def timestep(step_count, reward_record, exps):
    rews = []
    states = env.construct_state()
    np_states = np.array(states)

    actions = []

    # choose action for each vehicle
    for v_index in range(num_vehicles):
        action = choose_action(dqns[v_index], np_states[v_index: v_index + 1])
        actions.append(action)

    for v_index in range(num_vehicles):
        # convert action from scalar representation into vector representation
        action_vector = [int(actions[v_index] / num_servers) * alloc_unit, actions[v_index] % num_servers + 1]
        rews.append(env.calculate_reward(v_index, action_vector))
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
            exp_mems[v_index].append(exps[v_index][index] + [w_sum])
        # experience is in the format:
        # [state, action, reward, next state, weighted reward sum]
        return w_sum


def update_parameters():
    rand_inds = random.sample(range(len(exp_mems[0])), batch_size)  # random indices for experiences
    for v_index in range(num_vehicles):
        exps_batch = [exp_mems[v_index][rand_ind] for rand_ind in rand_inds]
        states_batch = [rand_exp[0] for rand_exp in exps_batch]  # shape: (batch_size, input_size)
        actions_batch = [rand_exp[1] for rand_exp in exps_batch]  # shape: (batch_size,)
        WRSs_batch = [rand_exp[4] for rand_exp in exps_batch]  # WRS means weighted reward sum. shape: (batch_size,)

        np_states_batch = np.array(states_batch)
        dqn = dqns[v_index]
        WRSs_batch = tf.convert_to_tensor(WRSs_batch)

        with tf.GradientTape() as tape:
            qualities_batch = dqn(
                np_states_batch)  # batch of qualities of all possible actions. shape: (batch_size, output_size)
            action_q_batch = [qualities_batch[i][actions_batch[i]] for i in
                              range(batch_size)]  # batch of quality of selected action. shape: (batch_size, )
            # action_q_batch = tf.convert_to_tensor(action_q_batch, dtype=tf.float64)
            action_q_batch = tf.convert_to_tensor(action_q_batch, dtype=WRSs_batch.dtype)
            loss = sum(abs(action_q_batch - WRSs_batch)) / batch_size

        grads = tape.gradient(loss, dqn.trainable_variables)
        optim = optims[v_index]
        optim.apply_gradients(zip(grads, dqn.trainable_variables))

    return loss


step_count = 1
optims = [tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(0.01, 50, 0.8))
          for i in range(num_vehicles)]  # each vehicle has one optimizer

for i in range(num_episodes):
    reward_record = []
    exps = [[] for i in range(num_vehicles)]
    env = Env(num_vehicles, num_servers, 'train.csv', 'simulated_tasks.csv')  # initialize env (set timestep as 1)
    print(f'Starting episode {i}')
    for j in range(num_timesteps):
        w_sum = timestep(step_count, reward_record, exps)
        loss = None
        if len(exp_mems[0]) >= batch_size:
            loss = update_parameters()

        # if step_count % 100 == 0:
        NUM = 50
        if step_count % NUM == 0:
            print(f'step: {step_count}')
            print(f'loss: {loss}')
            print(f'WRS: {w_sum}')
            print(f'reward sum: {sum(reward_record[-NUM:])}')
            print()

        step_count += 1
