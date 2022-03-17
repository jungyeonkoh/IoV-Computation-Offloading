from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
import environment as envs
from A3C_model import ActorCritic
from test import test
from train import train

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=16,
                    help='how many training processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=64,
                    help='number of forward steps in A3C (default: 64)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='offloading',
                    help='environment to train on (default: offloading)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--nv',default=100,
                    help='number of vehicles. (default: 100)')
parser.add_argument('--ns',default=12,
                    help='number of servers. (default: 12)')
parser.add_argument('--load_vehicle_position',default='./train.csv',
                    help='number of vehicles. (default: ./train.csv)')
parser.add_argument('--load_task_position',default='./simulated_tasks.csv',
                    help='number of vehicles. (default: ./simulated_tasks.csv)')
parser.add_argument('--load_vehicle_position_test',default='./test.csv',
                    help='number of vehicles. (default: ./test.csv)')
parser.add_argument('--load_task_position_test',default='./simulated_tasks_test.csv',
                    help='number of vehicles. (default: ./simulated_tasks_test.csv)')
parser.add_argument('--train_step',default=22359,
                    help='number of servers. (default: 22359)')
parser.add_argument('--test_step',default=11939,
                    help='number of servers. (default: 11939)')



if __name__ == '__main__':

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    shared_model = ActorCritic(
        26, args.ns)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
