
import bz2
import copy
import pickle
import numpy as np
import torch
from torchrl.envs.common import EnvBase
from torchrl.data import Composite, Unbounded, Categorical
from tensordict import TensorDict

from src.Environment.Reward import *
from src.Environment.State import *
from src.Environment.Actions import *
from src.Environment.Vizualization import *


class TorchRLEnvironmentWrapper(EnvBase):
    def __init__(self, env):
        super().__init__()
        self.env = env
        # Set observation and action specs
        self.observation_spec = self._make_observation_spec()
        self.action_spec = self._make_action_spec()
        self.reward_spec = self._make_reward_spec()
        self._batch_size = torch.Size([])  # single environment, not vectorized


    def _reset(self, tensordict=None):
        obs, info = self.env.reset(training=True)
        return self.build_tensordict(obs, info)

    def _step(self, tensordict):
        actions = tensordict["agents"]["action"]  # shape: [num_agents]
        obs, reward, terminated, truncated, info = self.env.step(actions.tolist())
        td= self.build_tensordict(obs, info, reward, terminated, truncated)
        return td

    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def build_tensordict(self, obs, info, reward=None, terminated=None, truncated=None):
        state_array, t_to_go, last_action, out_of_bounds = obs
        td = TensorDict({}, batch_size=[])
        # per-agent observations
        td.set(("agents", "observation"), torch.tensor(state_array, dtype=torch.float32))
        #td.set(("agents", "t_to_go"), torch.tensor(t_to_go, dtype=torch.float32))
        #td.set(("agents", "last_action"), torch.tensor(last_action, dtype=torch.int64))
        #td.set(("agents", "out_of_bounds"), torch.tensor(out_of_bounds, dtype=torch.float32))
        if reward is not None:
            td.set(("agents","reward"), torch.tensor(np.array(reward), dtype=torch.float32))
        if terminated is not None:
            td.set(("terminated"), torch.tensor(terminated, dtype=torch.bool))
        #if truncated is not None:
         #   td.set(( "truncated"), torch.tensor(truncated, dtype=torch.bool))
        #if terminated is not None:
            done = np.logical_or(terminated, truncated)
            td.set(("done"), torch.tensor(done, dtype=torch.bool))
        return td

    def _make_observation_spec(self):
        obs, _ = self.env.reset(training=False)
        state_array = obs[0]  # shape [n_agents, C, H, W]
        n_agents, F, C, H, W = state_array.shape
        return Composite({
            ("agents", "observation"): Unbounded((n_agents, F, C, H, W), dtype=torch.float32)
        })


    def _make_action_spec(self):
        n = self.env.params.max_number_agents
        return Composite({("agents", "action"): Categorical(len(Actions), shape=(n,), dtype=torch.int64)})


    def _make_reward_spec(self):
        n=self.env.params.max_number_agents
        return Composite({("agents","reward"): Unbounded((n,), dtype=torch.float32)})

