import os
import sys
from pathlib import Path

import torch
import hydra
from matplotlib import pyplot as plt

import wandb
from omegaconf import DictConfig, OmegaConf
from tensordict.nn import TensorDictSequential
from torch import nn
from torch.nn import Sequential, Flatten
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv, RewardSum
from torchrl.modules.tensordict_module.common import TensorDictModule
from torchrl.objectives import QMixerLoss, ValueEstimators, SoftUpdate, DQNLoss
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs.utils import set_exploration_type, ExplorationType, check_env_specs
from torchrl.modules import MLP, QValueModule, SafeSequential, MultiAgentConvNet, QValueActor
from torchrl.modules.models.multiagent import QMixer, MultiAgentMLP, VDNMixer
from torchrl.modules.tensordict_module.exploration import EGreedyModule
from torchrl._utils import logger as torchrl_logger
from hydra.utils import to_absolute_path


from src.Environment.TorchRLWrapper import *
from src.Environment.Environment import *
from src.Rainbow.model import MyNet


def save_memory(memory, memory_path):
    with open(memory_path, 'wb') as pickle_file:
        pickle.dump(memory, pickle_file)


@hydra.main(version_base="1.1", config_path="configs", config_name="eval")
def train(cfg: DictConfig):
    sys.setrecursionlimit(10000)
    # device setup
    cfg.model.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.model.device
    with open(to_absolute_path('configs/eval_general.yaml'), 'r') as f:
        conf = yaml.safe_load(f)

    # --- Build environments ---
    env = Environment(EnvironmentParams(conf['env1']))
    env = TorchRLEnvironmentWrapper(env)
    check_env_specs(env)

    # --- Build network ---
    n, F, C, H, W = env.observation_spec[('agents','observation')].shape
    act_spec = env.action_spec[('agents','action')]
    net = MyNet(n_agents=n,centralized=False,share_params=True,device=cfg.model.device)
    module = TensorDictModule(
        net, in_keys=[("agents","observation"),("agents","t_to_go"),("agents","last_action"),("agents","out_of_bounds")], out_keys=[("agents", "action_value")])

    value_module = QValueModule(
        action_value_key=("agents", "action_value"),
        out_keys=[
            env.action_key,
            ("agents", "action_value"),
            ("agents", "chosen_action_value"),
        ],
        spec=env.full_action_spec_unbatched,
        action_space=None,
    )
    qnet = SafeSequential(module, value_module)

    # --- Load pretrained model if available ---
    if getattr(cfg.model, 'pretrained_qnet_path', None):
        qnet_path = to_absolute_path(cfg.model.pretrained_qnet_path)
        if os.path.isfile(qnet_path):
            state = torch.load(qnet_path, map_location=cfg.model.device)
            missing_keys, unexpected_keys = qnet.load_state_dict(state, strict=False)
            net = qnet[0]
            torchrl_logger.info(f"Loaded qnet from {qnet_path}")
        else:
            torchrl_logger.warn(f"QNet path {qnet_path} not found.")


    T_overlap, not_finished, T_timesave = [[] for _ in range(51)], [0 for _ in range(51)], [[] for _ in range(51)]
    env_args = conf['env1']
    print(qnet[0].n_agents)

    for n in range(5,6):
        for size in range(10, 51):
            print(size)
            env_args['dataset_path'] =  to_absolute_path('maps/datasets/multi_agent10%/' + str(n) + '_' + str(size) + '.pth')
            env = Environment(EnvironmentParams(env_args))
            env = TorchRLEnvironmentWrapper(env)
            env.env.env_params.state_ptr = 0
            done = True
            truncated = False
            #env.env.rendering=True
            #env.rollout(10000,qnet)
            for t in range(50):

                while True:
                    if done or truncated:
                        env.env.params.number_agents = n
                        td = env.reset(training=False)
                        env.env.rewards.reset(env.env.state)
                        env.env.remaining = env.env.state.remaining
                        env.env.heuristic_position = [None for _ in range(env.env.params.number_agents)]
                        reward_sum, done, truncated = 0, False, False
                    action = qnet(td)
                    a = action["agents"].get("action").cpu().numpy()
                    info = env.env.get_info(3, 8)
                    info = env.env.filter(a, info)
                    for i in range(len(info)):
                        if a[i] == 4:
                            info[i] = True
                    if any(info):
                        #action =  qnet(td)
                        ac = env.env.get_heuristic_action(info)
                        for i, a in enumerate(ac):
                            if a is not None:
                                action["agents"]["action"][i] = torch.tensor(a)
                            if  action["agents"]["action"][i] == 4 and info[i] == False:
                                print(action, info, env.env.state.remaining)
                    td = env.step(action)  # Step
                    td = td["next"]
                    done = env.env.state.terminated
                    truncated = env.env.state.truncated
                    if done or truncated:
                        if not truncated:
                            print("env: ", size, " episode ", t, " time_save: ", env.env.rewards.get_time_save())
                            T_timesave[size - 5].append(env.env.rewards.get_time_save())
                            T_overlap[size - 5].append(env.env.rewards.get_overlap())
                        else:
                            not_finished[size - 5] += 1
                            T_timesave[size - 5].append(env.env.rewards.get_time_save())
                            print("env: ", size, " episode ", t, " not finished")
                        break

        print(not_finished)
        path = to_absolute_path(cfg.eval.path)
        Path(path).mkdir(exist_ok=True)
        save_memory(T_overlap, path + '/overlap.pkl')
        save_memory(T_timesave, path + '/timesave.pkl')
        save_memory(not_finished, path + '/not_finished.pkl')
    fig = plt.figure(figsize=(10, 7))

    # Creating plot
    plt.boxplot(np.asarray(T_timesave, dtype="object"))

    # show plot
    plt.show()

if __name__ == "__main__":
    train()