import time
import copy

from numpy import ceil

from src.Environment.Actions import Actions, Events
import random
import numpy as np
import yaml
from src.Environment.Grid import GridMap


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_position(self):
        return self.x, self.y


class StateParams:
    def __init__(self, args):
        self.max_number_agents = args['number_agents']
        self.random_number_agents =args['random_number_agents']
        self.number_agents =self.max_number_agents
        self.size = args['size']
        self.min_size = args['min_size']
        self.random_size = args['random_size']
        self.obstacles_random = args['obstacles_random']
        self.number_obstacles = args['number_obstacles']
        self.starting_position_random = args['starting_position_random']
        self.starting_position = args['starting_position']
        self.starting_position_corner = args['starting_position_corner']
        self.real_size = None
        self.sensor_range = args['sensor_range']
        self.sensor = args['sensor']
        self.random_coverage = args['random_coverage']
        if args['map_config'] != 'empty':
            with open(args['map_config'], 'r') as file:
                yaml_data = yaml.safe_load(file)
            self.map_data = np.array(yaml_data['map'])
        else:
            self.map_data = None


class State:
    def __init__(self, Params):
        self.params = Params
        self.local_map: GridMap = None
        self.global_map = None
        self.position = [Position(-1, - 1) for _ in range(self.params.number_agents)]
        self.remaining = None
        self.last_action = [[4 for _ in range(3)] for _ in range(self.params.number_agents)]
        self.out_of_bounds = [[None for _ in range(3)] for _ in range(self.params.number_agents)]
        self.timesteps = None
        self.t_to_go = [None for _ in range(self.params.number_agents)]
        self.optimal_steps = None
        self.terminated = False
        self.truncated = False
        self.state_array = [[None for _ in range(3)] for _ in range(self.params.number_agents)]
        self.last_positions = [[None for _ in range(4)] for _ in range(self.params.number_agents)]

    def move_agent(self, actions):
        events = [[] for _ in range(self.params.number_agents)]
        for i, action in enumerate(actions):
            old_x = self.position[i].x
            old_y = self.position[i].y
            other_agents = []
            for j in range(len(actions)):
                if i != j:
                    other_agents.append((self.position[j].get_position()))
                else:
                    other_agents.append(None)
            self.position[i].x += -1 if action == Actions.NORTH else 1 if action == Actions.SOUTH else 0
            # Change the column: 1=right (+1), 3=left (-1)
            self.position[i].y += 1 if action == Actions.EAST else -1 if action == Actions.WEST else 0
            blocked = False
            # if action == Actions.WAIT:
            #     events[i].append(Events.WAITED)
            if self.position[i].x < 0:
                self.position[i].x = 0
                blocked = True
            elif self.position[i].x >= self.local_map.height:
                self.position[i].x = self.local_map.height - 1
                blocked = True
            if self.position[i].y < 0:
                self.position[i].y = 0
                blocked = True
            elif self.position[i].y >= self.local_map.width:
                self.position[i].y = self.local_map.width - 1
                blocked = True
            elif (self.position[i].x, self.position[i].y) in self.local_map.obstacle_list:
                self.position[i].x = old_x
                self.position[i].y = old_y
                blocked = True
            elif (self.position[i].x, self.position[i].y) in other_agents:
                for j in range(len(actions)):
                    if i != j and (self.position[i].get_position() == other_agents[j]):
                        events[j].append(Events.BLOCKED)
                self.position[i].x = old_x
                self.position[i].y = old_y
                blocked = True
            if blocked:
                events[i].append(Events.BLOCKED)
                self.position[i].x = old_x
                self.position[i].y = old_y

            if self.position[i].get_position() not in self.local_map.visited_list and not blocked:
                self.local_map.visit_tile((self.position[i].x, self.position[i].y))
                self.remaining -= 1
                events[i].append(Events.NEW)

            self.local_map.update_agent_position((old_x, old_y), (self.position[i].x, self.position[i].y))
            self.last_action[i].append(action.value)
            self.last_action[i].pop()
            if self.params.sensor == "laser":
                self.local_map.laser_scanner(self.position[i].get_position(), self.global_map, self.params.sensor_range)
            elif self.params.sensor == "camera":
                self.local_map.camera(self.position[i].get_position(), self.global_map, self.params.sensor_range)
            self.local_map.update_agent_position((old_x, old_y), (self.position[i].x, self.position[i].y))
            self.state_array[i].pop(0)
            self.out_of_bounds[i].pop(0)
            self.last_positions[i].pop(0)
            s, o = self.local_map.center_map(self.position[i].get_position())
            self.state_array[i].append(s)
            self.out_of_bounds[i].append(o)
            self.last_positions[i].append(self.position[i].get_position())
            self.timesteps += 1
            self.t_to_go[i] -= 1
        if self.t_to_go[0] <= 0:
            self.truncated = True
            events[i].append(Events.TIMEOUT)

        if self.remaining < 1:
            self.terminated = True
            events[i].append(Events.FINISHED)
        return events

    def init_episode(self):
        if self.params.random_number_agents:
            self.params.number_agents = np.random.randint(1,self.params.max_number_agents+1)
        else:
            self.params.number_agents=self.params.max_number_agents
        if self.params.random_size:
            width = random.randint(self.params.min_size, self.params.size)
            height = width
            self.params.real_size = width
        else:
            width = self.params.size
            height = width
            self.params.real_size = self.params.size

        self.position = [Position(-1, - 1) for _ in range(self.params.number_agents)]
        if self.params.starting_position_random:
            for i in range(self.params.number_agents):
                while True:
                    pos = Position(random.randint(0, height - 1), random.randint(0, width - 1))
                    if pos not in self.position:
                        if i > 0:
                            dist = [np.linalg.norm(np.array(pos.get_position()) - np.array(self.position[j].get_position())) for j in range(i)]
                            if min(dist) > self.params.real_size / (self.params.number_agents + 0.5):
                                self.position[i] = pos
                                break
                        else:
                            self.position[i] = pos
                            break
        else:
            self.position[0] = Position(self.params.starting_position[0], self.params.starting_position[1])

        if self.params.map_data is not None:
            mapa = self.params.map_data
        else:
            mapa = np.zeros((height, width), dtype=int)
            if self.params.number_obstacles > 0:
                if self.params.obstacles_random:
                    obstacle_number = random.randint(0, self.params.number_obstacles)
                else:
                    obstacle_number = self.params.number_obstacles
                obstacles = 0
                while obstacles != obstacle_number:
                    coord = Position(random.randint(0, height - 1), random.randint(0, width - 1))
                    if coord not in self.position:
                        mapa[coord.x, coord.y] = -1
                        obstacles += 1

        self.global_map = GridMap(mapa)

        if self.params.map_data is not None or self.params.number_obstacles > 0:
            for pos in self.position:
                self.global_map.fix_map(pos.get_position())
        if self.params.sensor == "full information":
            self.local_map = self.global_map
        else:
            self.local_map = GridMap(start=[self.position[0].get_position()])

        for pos in self.position:
            self.local_map.visit_tile(pos.get_position())
            self.local_map.update_agent_position(pos.get_position(), pos.get_position())

        if self.params.random_coverage and np.random.random() < 0.15 and self.params.sensor == "full information":
            for i in range(0, random.randint(0, ceil(self.params.real_size ** 2 / 1.5))):
                tile = (random.randint(0, self.params.real_size - 1), random.randint(0, self.params.real_size - 1))
                if tile not in self.local_map.visited_list and tile in set(self.global_map.getTiles()).difference(
                        self.global_map.obstacle_list):
                    self.local_map.visit_tile(tile)

        if self.params.sensor == "laser":
            for position in self.position:
                self.local_map.laser_scanner(position.get_position(), self.global_map, self.params.sensor_range)
        elif self.params.sensor == "camera":
            for position in self.position:
                self.local_map.camera(position.get_position(), self.global_map, self.params.sensor_range)

        self.remaining = len(set(self.global_map.getTiles()).difference(self.global_map.obstacle_list)) - len(
            self.local_map.visited_list)
        if self.remaining < 1:
            self.init_episode()
        self.optimal_steps = self.remaining
        self.timesteps = 0
        self.t_to_go = [self.params.size ** 2 * 5 for _ in range(self.params.number_agents)]
        self.terminated = False
        self.truncated = False
        self.last_action = [[4 for _ in range(3)] for _ in range(self.params.number_agents)]
        self.out_of_bounds = [[None for _ in range(3)] for _ in range(self.params.number_agents)]
        self.state_array = [[None for _ in range(3)] for _ in range(self.params.number_agents)]
        self.last_positions = [[None for _ in range(4)] for _ in range(self.params.number_agents)]
        for i, position in enumerate(self.position):
            s, o = self.local_map.center_map(position.get_position())
            self.state_array[i] = [s, s, s]
            self.last_action[i] = [4, 4, 4]
            self.out_of_bounds[i] = [o, o, o]
            self.last_positions[i] = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]

    def partial_reset(self):
        self.t_to_go = [self.params.size ** 2 * 2 for _ in range(self.params.number_agents)]
        self.terminated = False
        self.truncated = False
        self.timesteps = 0
        for i, position in enumerate(self.position):
            s, o = self.local_map.center_map(position.get_position())
            self.state_array[i].append(s)
            self.state_array[i].pop(0)
            self.out_of_bounds[i].append(o)
            self.out_of_bounds[i].pop(0)
            self.last_positions[i].append(position.get_position())
            self.last_positions[i].pop(0)
        if self.remaining < 1:
            self.init_episode()

    def init_from_map(self, mapa):

        self.local_map = None
        self.global_map = copy.deepcopy(mapa)
        self.position = [Position(-1, - 1) for _ in range(self.params.number_agents)]
        self.remaining = None
        self.last_action = [[4 for _ in range(3)] for _ in range(self.params.number_agents)]
        self.out_of_bounds = [[None for _ in range(3)] for _ in range(self.params.number_agents)]
        self.timesteps = None
        self.t_to_go = [None for _ in range(self.params.number_agents)]
        self.optimal_steps = None
        self.terminated = False
        self.truncated = False
        self.state_array = [[None for _ in range(3)] for _ in range(self.params.number_agents)]
        self.last_positions = [[None for _ in range(4)] for _ in range(self.params.number_agents)]
        self.params.real_size = max(self.global_map.height, self.global_map.width)
        for i in range(self.params.number_agents):
            while True:
                pos = Position(random.randint(0, self.global_map.height - 1),
                               random.randint(0, self.global_map.width - 1))
                if pos not in self.position and pos.get_position() not in self.global_map.obstacle_list and pos.get_position() in self.global_map.getTiles():
                    self.position[i] = pos
                    break

        if self.params.sensor == "full information":
            self.local_map = self.global_map
        else:
            self.local_map = GridMap(start=[self.position[0].get_position()])

        for pos in self.position:
            self.local_map.visit_tile(pos.get_position())
            self.local_map.update_agent_position(pos.get_position(), pos.get_position())

        if self.params.random_coverage and np.random.random() < 0.5 and self.params.sensor == "full information":
            for i in range(0, random.randint(0, ceil(self.params.real_size ** 2 / 1.5))):
                tile = (random.randint(0, self.params.real_size - 1), random.randint(0, self.params.real_size - 1))
                if tile not in self.local_map.visited_list and tile in set(self.global_map.getTiles()).difference(
                        self.global_map.obstacle_list):
                    self.local_map.visit_tile(tile)

        if self.params.sensor == "laser":
            for position in self.position:
                self.local_map.laser_scanner(position.get_position(), self.global_map, self.params.sensor_range)
        elif self.params.sensor == "camera":
            for position in self.position:
                self.local_map.camera(position.get_position(), self.global_map, self.params.sensor_range)
                    
        self.remaining = len(set(self.global_map.getTiles()).difference(self.global_map.obstacle_list)) - len(
            self.local_map.visited_list)
        self.optimal_steps = self.remaining
        self.timesteps = 0
        self.t_to_go = [self.params.size ** 2 * 5 for _ in range(self.params.number_agents)]
        self.terminated = False
        self.truncated = False
        for i, position in enumerate(self.position):
            s, o = self.local_map.center_map(position.get_position())
            self.state_array[i] = [s, s, s]
            self.last_action[i] = [4, 4, 4]
            self.out_of_bounds[i] = [o, o, o]
            self.last_positions[i] = [(-1,-1),(-1,-1),(-1,-1),(-1,-1)]
