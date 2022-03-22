from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.optim as optim

#import my_optim
import environment as envs
from Actor import Actor
from Critic import Critic
from test import test
from train import train

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--discount_rate', type=float, default=0.99,
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
parser.add_argument('--num-steps', type=int, default=256,
                    help='number of forward steps in A3C (default: 256)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='offloading',
                    help='environment to train on (default: offloading)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--nv',type=int,default=100,
                    help='number of vehicles. (default: 100)')
parser.add_argument('--ns',type=int,default=12,
                    help='number of servers. (default: 12)')
parser.add_argument('--load_vehicle_position',default='./train.csv',
                    help='number of vehicles. (default: ./train.csv)')
parser.add_argument('--load_task_position',default='./simulated_tasks.csv',
                    help='number of vehicles. (default: ./simulated_tasks.csv)')
parser.add_argument('--load_vehicle_position_test',default='./test.csv',
                    help='number of vehicles. (default: ./test.csv)')
parser.add_argument('--load_task_position_test',default='./simulated_tasks_test.csv',
                    help='number of vehicles. (default: ./simulated_tasks_test.csv)')
parser.add_argument('--train_step',type=int,default=22350,
                    help='number of servers. (default: 22350)')
parser.add_argument('--test_step',type=int,default=2000,
                    help='number of servers. (default: 11930)')
parser.add_argument('--hidden_layer_num',type=int,default=2,
                    help='number of hidden layer (default: 2)')
parser.add_argument('--hidden_dim_size',type=int,default=128,
                    help='number of hidden dimension size (default: 128)')
parser.add_argument('--batch_size',type=int,default=128,
                    help='number of batch size (default: 128)')

if __name__ == '__main__':

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    global_Actor = Actor(state_space=(args.ns*2)+2,
                  action_space=args.ns,
                  num_hidden_layer=args.hidden_layer_num,
                  hidden_dim=args.hidden_dim_size)
    global_Critic = Critic(state_space=(args.ns*2)+2,
                  num_hidden_layer=args.hidden_layer_num,
                  hidden_dim=args.hidden_dim_size)

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, global_Actor, counter))
    p.start()
    processes.append(p)


    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, global_Actor,global_Critic, counter, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
