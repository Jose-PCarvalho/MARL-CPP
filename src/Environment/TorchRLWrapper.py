
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
        self.state_spec = self._make_state_spec()
        #self._batch_size = torch.Size([])  # single environment, not vectorized
        self.device = torch.device('cuda:0')


    def _reset(self, tensordict=None,training=True):
        obs, info = self.env.reset(training=training)
        return self.build_tensordict(obs, info)

    def _step(self, tensordict):
        actions = tensordict["agents"]["action"]  # shape: [num_agents]
        obs, reward, terminated, truncated, info = self.env.step(actions.tolist())
        td= self.build_tensordict(obs, info, reward, terminated, truncated)
        return td

    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def build_tensordict(self, obs, info, reward=None, terminated=None, truncated=None,state=None):

        state_array, t_to_go, last_action, out_of_bounds = obs
        last_action = torch.nn.functional.one_hot(torch.tensor(last_action,dtype=torch.int64),5)
        n = self.env.params.max_number_agents
        td = TensorDict({})
        # per-agent observations
        agents_td =TensorDict({},batch_size=[n])
        agents_td.set(("observation"), torch.tensor(state_array, dtype=torch.float32))
        agents_td.set(("t_to_go"), torch.tensor(np.array(t_to_go).reshape((n,1)), dtype=torch.float32))
        agents_td.set(("last_action"), last_action)
        agents_td.set(("out_of_bounds"), torch.tensor(out_of_bounds, dtype=torch.float32))
        state = self.env.state.global_map.padded_map()
        td.set(("state"), torch.tensor(state,dtype=torch.float32))

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
        obs, _ = self.env.reset(training=False)
        state_array, t_to_go, last_action, out_of_bounds = obs
        n, F, C, H, W = state_array.shape
        observation_specs = []
        reward_specs = []
        last_action_specs = []
        out_of_bounds_specs = []
        for i in range(n):
            observation_specs.append(Unbounded(shape=(F,C,H,W),dtype=torch.float32))
            reward_specs.append(Unbounded(dtype=torch.float32))
            last_action_specs.append(Unbounded(shape=(3,5),dtype=torch.int64))
            out_of_bounds_specs.append(Unbounded(shape=(out_of_bounds.shape[-3],out_of_bounds.shape[-2],out_of_bounds.shape[-1]),dtype=torch.float32))

        observation_spec = Composite({
            "agents": Composite({
                "observation": torch.stack(observation_specs),
                "t_to_go": torch.stack(reward_specs),
                "last_action": torch.stack(last_action_specs),
                "out_of_bounds": torch.stack(out_of_bounds_specs),
            }, shape=(n,)),
            "state": Unbounded(shape=(4, 40, 40), dtype=torch.float32)
        })
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

    def _make_state_spec(self):
        return Composite({"state": Unbounded(shape=(4,40,40),dtype=torch.float32)})
