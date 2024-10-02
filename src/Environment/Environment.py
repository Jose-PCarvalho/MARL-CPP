import bz2
import copy
import pickle
import time

from torchgen.model import Return

from src.Environment.Reward import *
from src.Environment.State import *
from src.Environment.Actions import *
from src.Environment.Vizualization import *


def load_memory(memory_path):
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
        return pickle.load(zipped_pickle_file)


class EnvironmentParams:
    def __init__(self, args):
        self.state_params = StateParams(args)
        self.reward_params = RewardParams()
        if args['dataset_path'] != 'empty':
            self.dataset = load_memory(args['dataset_path'])
            print(args['dataset_path'], " loaded!")
        else:
            self.dataset = None
        self.load_state = args['load_state']
        self.load_random = args['load_random']
        self.state_ptr = 0


class Environment:
    def __init__(self, params: EnvironmentParams):
        self.rewards = GridRewards(params)
        self.state = State(params.state_params)
        self.episode_count = 0
        self.viz = Vizualization()
        self.params = params.state_params
        self.env_params = params
        self.stall_counter = 0
        self.remaining = 0  ## Only for stalling purposes, the actual variable is in the state.
        self.interesting_states = []
        self.was_partial = False
        self.heuristic_position = [None for _ in range(params.state_params.number_agents)]
        self.position_locked = [False for _ in range(params.state_params.number_agents)]

    def reset(self, training=True):

        if self.state.truncated and training:
            self.state.partial_reset()
            self.was_partial = True
        else:
            len_states = len(self.interesting_states)
            if not self.was_partial and len_states > 0 and training:
                self.interesting_states.pop(-1)
                len_states -= 1
            self.was_partial = False
            if len_states > 1 and np.random.random() < 0.5 and training:
                self.state = self.interesting_states.pop(0)
            else:
                if self.env_params.dataset is not None:
                    if self.env_params.load_random:
                        temp_state = copy.deepcopy(np.random.choice(self.env_params.dataset))
                    else:
                        temp_state = copy.deepcopy(self.env_params.dataset[self.env_params.state_ptr])
                        self.env_params.state_ptr += 1
                    if not self.env_params.load_state:
                        self.state.params.number_agents = self.params.number_agents
                        self.state.init_from_map(temp_state.global_map)
                    else :
                        self.state = temp_state
                else:
                    self.state.init_episode()

                if training:
                    self.interesting_states.append(copy.deepcopy(self.state))

        self.rewards.reset(self.state)
        self.remaining = self.state.remaining
        self.heuristic_position = [None for _ in range(self.state.params.number_agents)]
        self.position_locked = [False for _ in range(self.state.params.number_agents)]
        return self.get_observation(), self.get_info()

    def step(self, action):
        a = [Actions(ac) for ac in action]
        events = self.state.move_agent(a)
        reward = self.rewards.compute_reward(events, self.state)
        return self.get_observation(), reward, self.state.terminated, self.state.truncated, self.get_info()

    def action_space(self):
        return len(Actions)

    def render(self, center=False):
        if center:
            for i in range(self.state.params.number_agents):
                obs = self.get_observation()
                self.viz.render_center(obs[0][i][-1, :, :, :])
        else:
            self.viz.render(self.state.local_map.map_array)

    def get_observation(self):
        oob = np.array(self.state.out_of_bounds)
        oob = oob[:, :, :, 0:2]
        return (np.array(self.state.state_array), self.state.t_to_go, np.array(self.state.last_action), oob)

    def get_info(self,Ks=4,K=8):
        if self.state.remaining <= self.state.params.number_agents:
            return [True for i in range(self.state.params.number_agents)]
        small_stuck = [self.rewards.stuck[i] > Ks for i in range(self.state.params.number_agents)]
        stuck = [False for i in range(self.state.params.number_agents)]
        for i , s in enumerate(small_stuck):
            if s:
                current_pos = np.array(self.state.last_positions[i][-1])
                oldest_pos = np.array(self.state.last_positions[i][0])
                dist = np.linalg.norm(current_pos-oldest_pos, ord=1)
                if dist<=1 :
                    stuck[i]= True

            if self.rewards.stuck[i] >K:
                stuck[i] =True
        for i, s in enumerate(stuck):
            if not s:
                self.position_locked[i] = False
                self.heuristic_position[i] = None

        return stuck

    def get_heuristic_action(self,info):
        actions = [None for _ in range(self.state.params.number_agents)]
        self.paths = [None for _ in range(self.state.params.number_agents)]
        for a in range(self.state.params.number_agents):
            if info[a]:
                pos = self.heuristic_position[a]
                if pos is not None:
                    pos = tuple(pos)
                if not self.position_locked[a] or pos is None or pos in self.state.local_map.visited_list or pos == self.state.position[a].get_position():
                    self.paths[a] = self.find_heuristic_position(a)
                else:
                    self.paths[a] = self.state.local_map.dijkstra_search(self.state.position[a].get_position(),(self.heuristic_position[a][0],self.heuristic_position[a][1]))

        for a in range(self.state.params.number_agents):
            if not info[a]:
                actions[a] = None
                continue
            diff = np.array(self.paths[a][0]) - np.array(self.state.position[a].get_position())
            diff = (diff[0], diff[1])
            if diff == (1, 0):
                actions[a]=Actions.SOUTH.value
            elif diff == (-1, 0):
                actions[a]=Actions.NORTH.value
            elif diff == (0, 1):
                actions[a]=Actions.EAST.value
            elif diff == (0, -1):
                actions[a]=Actions.WEST.value
            elif diff == (0, 0):
                actions[a]=Actions.WAIT.value
        return actions



    def find_heuristic_position(self,a):
        positions, indices = self.state.local_map.path_min_manhattan(self.state.position[a].get_position())
        path = None
        self.position_locked[a] = False
        for i in indices:
            if tuple(positions[i]) not in self.heuristic_position:
                path = self.state.local_map.dijkstra_search(self.state.position[a].get_position(),
                                                        (positions[i][0], positions[i][1]))
                if len(path) <1:
                    continue
                self.position_locked[a] = True
                self.heuristic_position[a] = tuple(positions[i])
                break
            else:
                a2 = self.heuristic_position.index(tuple(positions[i]))
                path = self.contested_path(a,a2,positions[i])
                if path is not None:
                    break
        if path is None:
            path = [self.state.position[a].get_position()]
            self.position_locked[a] = True
            self.heuristic_position[a] = tuple(self.state.position[a].get_position())
        return path
    def contested_path(self,a1,a2,p):
        p1 = self.state.local_map.dijkstra_search(self.state.position[a1].get_position(),
                                                        (p[0], p[1]))
        p2 = self.state.local_map.dijkstra_search(self.state.position[a2].get_position(),
                                                  (p[0], p[1]))
        if len(p1) < len(p2):
            self.position_locked[a1] = True
            self.heuristic_position[a1] = tuple(p)
            self.heuristic_position[a2] = None
            self.position_locked[a2] = None
            self.paths[a2] = self.find_heuristic_position(a2)
            return p1
        return None



    def detect_collision(self, action,info):
        info_ = copy.deepcopy(info)
        a = [Actions(ac) for ac in action]
        for i in range(self.state.params.number_agents):
            next_p = copy.deepcopy(self.state.position[i])
            action = a[i]
            next_p.x += -1 if action == Actions.NORTH else 1 if action == Actions.SOUTH else 0
            next_p.y += 1 if action == Actions.EAST else -1 if action == Actions.WEST else 0
            ac = None
            if ((next_p.x, next_p.y) not in set(self.state.local_map.getTiles()).difference(set(self.state.local_map.obstacle_list))) or info[i] == False:
                info_[i] = True
        ac = self.get_heuristic_action(info_)
        return ac , info_