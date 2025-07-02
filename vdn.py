import os
import sys

import torch
import hydra
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


@hydra.main(version_base="1.1", config_path="configs", config_name="vdn")
def train(cfg: DictConfig):
    sys.setrecursionlimit(10000)
    # device setup
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = 40000 #cfg.collector.frames_per_batch

    # initialize W&B
    wandb.init(
        project="qmix-experiments",
        name=cfg.get("wandb", {}).get("run_name", None),
        resume="auto",
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    with open(to_absolute_path('configs/training_obstacles.yaml'), 'r') as f:
        conf = yaml.safe_load(f)
    wandb.config.update(conf, allow_val_change=True)

    # --- Build environments ---
    env = Environment(EnvironmentParams(conf['env1']))
    env = TorchRLEnvironmentWrapper(env)
    test_env = Environment(EnvironmentParams(conf['env1']))
    test_env = TorchRLEnvironmentWrapper(test_env)
    check_env_specs(env)

    # --- Build network ---
    n, F, C, H, W = env.observation_spec[('agents','observation')].shape
    act_spec = env.action_spec[('agents','action')]

    net = MyNet(n_agents=n,centralized=False,share_params=True,device=cfg.train.device)
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
    if getattr(cfg.train, 'pretrained_qnet_path', None):
        qnet_path = to_absolute_path(cfg.train.pretrained_qnet_path)
        if os.path.isfile(qnet_path):
            state = torch.load(qnet_path, map_location=cfg.train.device)
            qnet.load_state_dict(state)
            torchrl_logger.info(f"Loaded qnet from {qnet_path}")
        else:
            torchrl_logger.warn(f"QNet path {qnet_path} not found.")



    mixer = TensorDictModule(
        module=VDNMixer(
            n_agents=n,
            device=cfg.train.device,
        ),
        in_keys=[("agents", "chosen_action_value")],
        out_keys=["chosen_action_value"],
    )
    # mixer = TensorDictModule(
    #     module=QMixer(
    #         state_shape=env.state_spec_unbatched["state"].shape,
    #         mixing_embed_dim=32,
    #         n_agents=n,
    #         device=cfg.train.device,
    #     ),
    #     in_keys=[("agents", "chosen_action_value"), "state"],
    #     out_keys=["chosen_action_value"],
    # )
    if getattr(cfg.train, 'pretrained_mixer_path', None):
        mix_path = to_absolute_path(cfg.train.pretrained_mixer_path)
        if os.path.isfile(mix_path):
            mix_state = torch.load(mix_path, map_location=cfg.train.device)
            mixer.module.load_state_dict(mix_state)
            torchrl_logger.info(f"Loaded mixer from {mix_path}")
        else:
            torchrl_logger.warn(f"Mixer path {mix_path} not found.")

    # Loss and updater
    loss_module = QMixerLoss(qnet,mixer,delay_value=True,action_space=act_spec)
    loss_module.set_keys(
        action_value=('agents','action_value'),
        local_value=('agents','chosen_action_value'),
        global_value='chosen_action_value',
        action=env.action_key,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=0.99)
    target_net_updater = SoftUpdate(loss_module, eps=1 - cfg.loss.tau)

    # Collector & replay buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)
    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    e = 1
    number_envs = 7
    while e <= number_envs:
        print(f"Starting Environment {e}")
        env_str = 'env'+str(e)
        env = Environment(EnvironmentParams(conf[env_str]))
        env = TorchRLEnvironmentWrapper(env)
        test_env = Environment(EnvironmentParams(conf[env_str]))
        test_env = TorchRLEnvironmentWrapper(test_env)
        frames = conf[env_str]['base_steps']
        eps_init = 0.8
        if e>1:
            eps_init = 0.8
        qnet_explore = TensorDictSequential(
            qnet,
            EGreedyModule(
                eps_init= eps_init,
                eps_end=0.05,
                annealing_num_steps=int(frames * (1 / 2)),
                action_key=env.action_key,
                spec=env.full_action_spec_unbatched,
                device=cfg.env.device
            ),
        )

        collector = SyncDataCollector(env,
            qnet_explore,
            device=cfg.env.device,
            storing_device=cfg.train.device,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=frames,
        )
        for i, tensordict_data in enumerate(collector):
            #torchrl_logger.info(f"\nIteration {i}")
            sampling_time = time.time() - sampling_start
            # Remove agent dimension from reward (since it is shared in QMIX/VDN)
            tensordict_data.set(("next", "reward"), tensordict_data.get(("next", env.reward_key)).mean(-2))
            del tensordict_data["next", env.reward_key]

            current_frames = tensordict_data.numel()
            total_frames += current_frames
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            training_tds = []
            training_start = time.time()
            if total_frames > 20000:
                for _ in range(cfg.train.num_epochs):
                    for _ in range(1):#cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                        subdata = replay_buffer.sample()
                        loss_vals = loss_module(subdata)
                        training_tds.append(loss_vals.detach())
                        loss_value = loss_vals["loss"]
                        loss_value.backward()
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            loss_module.parameters(), cfg.train.max_grad_norm
                        )
                        training_tds[-1].set("grad_norm", total_norm.mean())
                        optim.step()
                        optim.zero_grad()
                        target_net_updater.step()

                qnet_explore[1].step(frames=current_frames)  # Update exploration annealing
                collector.update_policy_weights_()

                training_time = time.time() - training_start
                iteration_time = sampling_time + training_time
                total_time += iteration_time
                training_tds = torch.stack(training_tds)
                mean_loss = torch.stack([td["loss"] for td in training_tds]).mean().item()
                mean_grad = torch.stack([td["grad_norm"] for td in training_tds]).mean().item()
                wandb.log({
                    "train/iteration": i,
                    "train/loss": mean_loss,
                    "train/grad_norm": mean_grad,
                    "train/sampling_time": sampling_time,
                    "train/training_time": training_time,
                }, step=int(total_frames/4))

                if cfg.eval.evaluation_episodes > 0 and i % cfg.eval.evaluation_interval == 0:
                    with torch.no_grad():
                        # test_env.frames = []
                        test_env.env.rendering = False
                        rewards = []
                        time_save = []
                        overlap = []
                        for _ in range(cfg.eval.evaluation_episodes):
                            test_env.rollout(max_steps=10000,
                                             policy=qnet_explore,
                                             auto_cast_to_device=True,
                                             break_when_any_done=True)
                            rewards.append(test_env.env.rewards.get_cumulative_reward())
                            time_save.append(test_env.env.rewards.get_time_save())
                            #overlap.append(test_env.env.rewards.get_overlap())
                            test_env.reset()
                        mean_reward = float(torch.tensor(rewards).mean())
                        mean_time_save = float(torch.tensor(time_save).mean())
                        #mean_overlap = float(torch.tensor(overlap).mean())

                    print(f"Eval @ {env_str} iter {i*4}/{frames}: mean reward = {mean_reward:.3f},  mean time save = {mean_time_save:.3f},")
                    wandb.log({"eval/mean_reward": mean_reward}, step=int(total_frames/4))
                    wandb.log({"eval/mean_time_save": mean_time_save}, step=int(total_frames / 4))
                    #wandb.log({"eval/mean_overlap": mean_overlap}, step=int(total_frames / 4))

                    #log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)


        # --- Save final model ---
        cwd = os.getcwd()
        qnet_file = os.path.join(cwd, env_str+'_qnet_model.pth')
        mixer_file = os.path.join(cwd, env_str+'mixer_model.pth')
        torch.save(qnet.state_dict(), qnet_file)
        torch.save(mixer.module.state_dict(), mixer_file)
        wandb.save(qnet_file)
        wandb.save(mixer_file)
        torchrl_logger.info(f"Saved final model to {qnet_file}")
        e+=1

    # cleanup
    collector.shutdown(); env.close(); wandb.finish()

if __name__ == "__main__":
    train()