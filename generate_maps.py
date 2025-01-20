import bz2
import copy
import pickle
import random
import sys
from tqdm import trange
import os
from src.Environment.Environment import *
import yaml

with open('configs/map_generation.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file


def save_memory(memory, memory_path):
    if os.path.exists(memory_path):
        os.remove(memory_path)
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
        pickle.dump(memory, zipped_pickle_file)

def load_memory(memory_path):
    with bz2.open(memory_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

sys.setrecursionlimit(11000)
env = Environment(EnvironmentParams(conf['env1']))

# for n in range(1,2):
#     for s in range(5,51):
#         maps = []
#         env.state.params.size = s
#         print(n,s)
#         for T in trange(0, 50):
#             env.reset(False)
#             #env.render()
#             #time.sleep(1)
#             maps.append(copy.deepcopy(env.state))
#         save_memory(maps, 'maps/datasets/multi_agent5%/'+str(n)+'_'+str(s)+'.pth')


for n in range(2,16):
    for s in range(5,51):
        print(n,s)
        maps_=load_memory('maps/datasets/multi_agent5%/'+str(n-1)+'_'+str(s)+'.pth')
        maps = []
        for T in trange(0, 50):
            env.state = copy.deepcopy(maps_[T])
            #env.render()
            #time.sleep(1)
            env.state.add_one_agent()
            #env.render()
            #time.sleep(1)
            maps.append(copy.deepcopy(env.state))
        save_memory(maps, 'maps/datasets/multi_agent5%/'+str(n)+'_'+str(s)+'.pth')

