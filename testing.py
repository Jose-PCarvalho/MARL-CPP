import argparse
import bz2
import random
from datetime import datetime
import os
import pickle

import numpy as np
import torch
import yaml
from torch.nn import Sequential, Flatten
from torchrl.envs import TransformedEnv, RewardSum
from torchrl.modules import MultiAgentConvNet, MultiAgentMLP
from tqdm import trange
from src.Environment.Environment import *
from src.Rainbow.agent import *
from src.Rainbow.memory import ReplayMemory
from test import test
from src.Environment.TorchRLWrapper import *
import sys

sys.setrecursionlimit(11000)
def log(s, log_dir=None):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)
    if dir is not None:
        with open(log_dir, 'a') as file:
            file.write(('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s + '\n'))


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)




with open('configs/training_obstacles.yaml' , 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file

env = Environment(EnvironmentParams(conf['env1']))

torchrl_env = TorchRLEnvironmentWrapper(env)





# Reset environment
td = torchrl_env.reset()
print("Initial Observation:", td)
# Run for a few random steps
for step in range(5):
    # Sample random actions tensor nested under ('agents','action')
    action_spec = torchrl_env.action_spec
    random_actions = action_spec[('agents', 'action')].rand()  # correct usage
    print("Sampled actions:", random_actions)
    act = TensorDict({('agents', 'action'): random_actions}, batch_size=[])
    print("Sampled actions:", act)
    td = torchrl_env.step(act)
    print(td)
    print(f"Step {step+1}:")
    print("  Actions:", random_actions)
    print("  Reward:", td["next","agents"].get("reward"))
    print("  Done:", td["next"].get("done"))
    obs = td["next"].get("observation")
    if td["next"].get("done").all():
        print("Episode finished.")
        break

torchrl_env.env.rendering=False
td = torchrl_env.rollout(10000)
print(torchrl_env.env.rewards.cumulative_reward)
td=torch.zeros_like(td)
#print(td)


n, F, C, H, W = torchrl_env.observation_spec[('agents', 'observation')].shape
obs_shape = F * C * H * W
act_spec = torchrl_env.action_spec[('agents', 'action')]
act_dim = act_spec.n

test = np.zeros(torchrl_env.observation_spec[('agents', 'observation')].shape)
test = torch.tensor(test, dtype=torch.float32)
conv_net = MultiAgentConvNet(
    n,
    in_features=12,
    num_cells=[32, 64, 64],
    kernel_sizes=[3, 3, 3],
    strides=[1, 2, 2],
    paddings=[1, 1, 1],
    share_params=True,
    activation_class= torch.nn.ReLU,
    centralized=False,
)
mlp = MultiAgentMLP(n_agent_inputs=7744,
                    n_agent_outputs=5,
                    n_agents=n,
                    share_params=True,
                    centralized= False,
                    depth=2,
                    num_cells=256)
net = Sequential(Flatten(start_dim=-4,end_dim=-3),conv_net)
net = Sequential(net,mlp)
test = net(test)
print(test)