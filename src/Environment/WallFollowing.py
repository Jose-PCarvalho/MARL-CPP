from src.Environment.Actions import *
import numpy as np
import random


class WallFollower:
    def __init__(self):
        self.size = None
        self.n_actions = None
        self.action_seq = []
        self.action_idx = 0
        self.center = None
        self.north = None
        self.south = None
        self.west = None
        self.east = None
        self.wall_list = []
        self.dirs = None
        self.counter = 0

    def select_action(self, obs):
        obstacle_dim = 1
        if len(self.wall_list) < 2:
            for d in self.dirs:
                if obs[obstacle_dim][d] == 1:
                    if d not in self.wall_list:
                        self.wall_list.append(d)
                        self.assign_actions()

        if len(self.wall_list) == 2:
            a = self.action_seq[self.counter % len(self.action_seq)]
            self.n_actions -= 1
            if self.n_actions == 1 and self.counter == 3:
                self.counter = 0
                self.size -= 2
                self.n_actions = self.size + 1
            if self.n_actions == 0:
                self.n_actions = self.size
                self.counter += 1

        action = self.action_seq[self.counter]

        return action

    def init(self, obs, size):
        obstacle_dim = 1
        self.center = obs.shape[1] // 2
        self.north = (self.center - 1, self.center)
        self.south = (self.center + 1, self.center)
        self.west = (self.center, self.center - 1)
        self.east = (self.center, self.center + 1)
        self.dirs = [self.north, self.south, self.west, self.east]
        self.counter = 0
        self.wall_list = []
        self.n_actions = size
        self.size = size - 1

        if obs[obstacle_dim][self.north] == 1:
            self.wall_list.append(self.north)
        if obs[obstacle_dim][self.south] == 1:
            self.wall_list.append(self.south)
        if obs[obstacle_dim][self.west]:
            self.wall_list.append(self.west)
        if obs[obstacle_dim][self.east]:
            self.wall_list.append(self.east)

        self.assign_actions()

    def assign_actions(self):
        if len(self.wall_list) == 2:
            if self.north in self.wall_list and self.east in self.wall_list:
                self.action_seq = random.choice([[Actions.SOUTH, Actions.WEST, Actions.NORTH, Actions.EAST],
                                                 [Actions.WEST, Actions.SOUTH, Actions.EAST, Actions.NORTH]])

            elif self.north in self.wall_list and self.west in self.wall_list:
                l1 = [Actions.SOUTH, Actions.EAST, Actions.NORTH, Actions.WEST]
                l2 = [Actions.EAST, Actions.SOUTH, Actions.WEST, Actions.NORTH]
                self.action_seq = random.choice([l1, l2])

            elif self.south in self.wall_list and self.east in self.wall_list:
                self.action_seq = random.choice([[Actions.NORTH, Actions.WEST, Actions.SOUTH, Actions.EAST],
                                                 [Actions.WEST, Actions.NORTH, Actions.EAST, Actions.SOUTH]])

            elif self.south in self.wall_list and self.west in self.wall_list:
                self.action_seq = random.choice([[Actions.NORTH, Actions.EAST, Actions.SOUTH, Actions.WEST],
                                                 [Actions.EAST, Actions.NORTH, Actions.WEST, Actions.SOUTH]])

        elif len(self.wall_list) == 1:
            if self.north in self.wall_list:
                self.action_seq = [random.choice([Actions.WEST, Actions.EAST])]

            elif self.west in self.wall_list:
                self.action_seq = [random.choice([Actions.SOUTH, Actions.NORTH])]

            elif self.east in self.wall_list:
                self.action_seq = [random.choice([Actions.NORTH, Actions.SOUTH])]

            elif self.south in self.wall_list:
                self.action_seq = [random.choice([Actions.WEST, Actions.EAST])]

        else:
            self.action_seq = [random.choice(list(Actions))]
