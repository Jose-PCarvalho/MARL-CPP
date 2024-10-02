import argparse
import bz2
import random
from datetime import datetime
import os
import pickle

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import trange
from src.Environment.Environment import *
from src.Rainbow.agent import *
from src.Rainbow.memory import ReplayMemory
from test import test

def log(s, log_dir=None):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)
    if dir is not None:
        with open(log_dir, 'a') as file:
            file.write(('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s + '\n'))


def load_memory(memory_path):
    with open(memory_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def save_memory(memory, memory_path):
    with open(memory_path, 'wb') as pickle_file:
        pickle.dump(memory, pickle_file)


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--id', type=str, default='CPP_EVALr', help='Experiment ID')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--T-max', type=int, default=int(70e4), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='data-efficient', choices=['canonical', 'data-efficient'],
                    metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-625, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=625, metavar='V', help='Maximum of value distribution support')

parser.add_argument('--model', type=str, metavar='PARAMS',
                    default='results/K025_new/checkpoint.pth',
                    help='Pretrained model (state dict)')

parser.add_argument('--memory-capacity', type=int, default=int(1), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.9999, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=4e-5, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(2e3), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=25000, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=100, metavar='N',
                    help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=2000, metavar='N',
                    help='Number of transitions to use for validating Q')

parser.add_argument('--render', action='store_true', default=False, help='Display screen (testing only)')

parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=1,
                    help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true',
                    help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--config-file', type=str, default='configs/eval_general.yaml')
parser.add_argument('--log-file', type=str, default='results/log.txt')
parser.add_argument('--starting-environment', type=int, default=1)
parser.add_argument('--tau', type=float, default=0.004)

# Setup
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf'), 'overlap': []}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
    print("Initiating cuda")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = True
else:
    args.device = torch.device('cpu')
    print("Initiating CPU")

with open(args.config_file, 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file

if args.log_file:
    with open(args.log_file, 'w') as file:
        pass

T_overlap, not_finished = [[] for _ in range(26)], [0 for _ in range(26)]

env_args = conf['env1']
for n in range(2,3):
    for size in range(10, 26):
        print(size)
        env_args['dataset_path'] = 'maps/datasets/eval/' + str(n) + '_' + str(size) + '.pth'
        env = Environment(EnvironmentParams(env_args))
        action_space = env.action_space()
        dqn = Agent(args, action_space)
        done = True
        truncated = False
        dqn.eval()


        for t in range(50):

            while True:

                if done or truncated:
                    state, info = env.reset(False)

                    env.params.number_agents=n
                    env.rewards.reset(env.state)
                    env.position_locked = [False for _ in range(env.params.number_agents)]
                    env.remaining = env.state.remaining
                    env.heuristic_position = [None for _ in range(env.params.number_agents)]
                    reward_sum, done, truncated = 0, False, False

                action = dqn.act(state[0], state[1], state[2], state[3])  # Choose an action ε-greedily

                for i in range(len(info)):
                    if action[i] == 4:
                        info[i] = True
                if any(info):
                    #action = dqn.act(state[0], state[1], state[2], state[3])
                    ac = env.get_heuristic_action(info)
                    
                    for i, a in enumerate(ac):
                         if a is not None:
                             action[i] = a
                         if action[i]== 4 and info[i]==False:
                             print(action,info, env.state.remaining)


                state, reward, done, truncated, info = env.step(action)  # Step

                if args.render:
                    env.render()
                if done or truncated:
                    if not truncated:

                        print("env: ", size, " episode ", t, " time_save: ", env.rewards.get_time_save())
                        T_overlap[size - 5].append(env.rewards.get_time_save())
                    else:

                        not_finished[size - 5] += 1
                    break

    print(not_finished)
    save_memory(T_overlap, 'stats/timesave_'+str(n)+'test2.pkl')
fig = plt.figure(figsize=(10, 7))

# Creating plot
plt.boxplot(T_overlap)

# show plot
plt.show()
