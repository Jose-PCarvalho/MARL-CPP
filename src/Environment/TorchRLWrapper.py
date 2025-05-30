
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
        self.done_spec = Categorical(n = 2,shape = torch.Size((1,)),dtype = torch.bool,)
        #self._batch_size = torch.Size([])  # single environment, not vectorized
        self.device = torch.device('cuda:0')


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

        n = self.env.params.max_number_agents
        td = TensorDict({})
        # per-agent observations
        agents_td =TensorDict({},batch_size=[n])
        agents_td.set(("observation"), torch.tensor(state_array, dtype=torch.float32))
        #td.set(("agents", "t_to_go"), torch.tensor(t_to_go, dtype=torch.float32))
        #td.set(("agents", "last_action"), torch.tensor(last_action, dtype=torch.int64))
        #td.set(("agents", "out_of_bounds"), torch.tensor(out_of_bounds, dtype=torch.float32))
        if reward is not None:
            agents_td.set(("reward"), torch.tensor(np.array(reward), dtype=torch.float32))

        #if terminated is not None:
            #td.set(("terminated"), torch.tensor(terminated, dtype=torch.bool))
        #if truncated is not None:
         #   td.set(( "truncated"), torch.tensor(truncated, dtype=torch.bool))
        if terminated is not None or truncated is not None:
            done = np.logical_or(terminated, truncated)
            td.set(("done"), torch.tensor(done, dtype=torch.bool))
        td.set("agents",agents_td)

        return td.to(self.device)

    def _make_observation_spec(self):
        # 1) grab one reset to infer shapes
        obs, _ = self.env.reset(training=False)
        #    obs[0] has shape [n_agents, F, C, H, W]
        state_array = obs[0]
        n, F, C, H, W = state_array.shape
        observation_specs = []
        for i in range(n):
            observation_specs.append(Unbounded(shape=(F,C,H,W),dtype=torch.float32))

        observation_spec = Composite({"agents": Composite({"observation": torch.stack(observation_specs)}, shape = (n,))})
        return observation_spec




    def _make_action_spec(self):
        n = self.env.params.max_number_agents
        # action_spec = Categorical(n=len(Actions),device="cuda:0", dtype=torch.int64,shape=torch.Size([n,]))
        # agents_spec = Composite(action=action_spec,device="cuda:0",shape=torch.Size([n,]))
        # spec = Composite(agents=agents_spec,device="cuda:0",shape=torch.Size([]))
        # print(spec)
        action_specs = []
        for i in range(n):
            action_specs.append(Categorical(n=len(Actions),device="cuda:0", dtype=torch.int64))
        action_spec = Composite(
            {
                "agents":Composite({"action":torch.stack(action_specs)},shape=(n,))
            })
        return action_spec


    def _make_reward_spec(self):
        n=self.env.params.max_number_agents
        reward_specs = []
        for i in range(n):
            reward_specs.append(Unbounded(dtype=torch.float32))
        reward_spec = Composite({"agents": Composite({"reward": torch.stack(reward_specs)},shape=(n,))})
        return reward_spec

