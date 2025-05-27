import torch
from torch.nn import Sequential, Flatten
from torchrl.collectors import SyncDataCollector
from torchrl.modules.tensordict_module.common import TensorDictModule
from torchrl.objectives import QMixerLoss, ValueEstimators, SoftUpdate
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs.utils import set_exploration_type, ExplorationType, check_env_specs
from torchrl.modules import MLP, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import QMixer
from torchrl.modules.tensordict_module.exploration import EGreedyModule

from src.Environment.TorchRLWrapper import *
from src.Environment.Environment import *

# Hyperparameters (since config has no collector section)
frames_per_batch = 32
total_frames = 3200
minibatch_size = 32
num_epochs = 4
learning_rate = 1e-3
max_iterations = 100

with open('configs/training_obstacles.yaml', 'r') as f:
    conf = yaml.safe_load(f)
raw_env = Environment(EnvironmentParams(conf['env1']))
env = TorchRLEnvironmentWrapper(raw_env)
check_env_specs(env)

# Flatten observation per agent: C x H x W
n, F, C, H, W = env.observation_spec[('agents','observation')].shape
obs_shape = F* C * H * W
act_spec = env.action_spec[('agents','action')]
act_dim = act_spec.n

policy = Sequential(
    Flatten(start_dim=-4),  # flatten per-agent C,H,W
    MLP(obs_shape, act_spec.n, depth=2, num_cells=256)
)
module = TensorDictModule(
    policy,
    in_keys=[('agents','observation')],
    out_keys=[('agents','action_value')]
)
qvalue_module = QValueModule(
    action_value_key=('agents','action_value'),
    out_keys=[('agents','action'), ('agents','action_value'), ('agents','chosen_action_value')],
    spec=act_spec,
    action_space=None
)
qnet = SafeSequential(module, qvalue_module)

# Exploration
exploration = EGreedyModule(
    eps_init=0.3,
    eps_end=0.05,
    annealing_num_steps=total_frames // 2,
    action_key=('agents','action'),
    spec=act_spec
)
qnet_explore = SafeSequential(qnet, exploration)

# Mixer
mixer = TensorDictModule(
    module=QMixer(
        state_shape=(obs_shape,),
        mixing_embed_dim=32,
        n_agents=conf['env1']['number_agents'],
        device='cuda:0',
    ),
    in_keys=[('agents','chosen_action_value'), ('agents','observation')],
    out_keys=['chosen_action_value']
)

# Loss and updater
loss_module = QMixerLoss(
    qnet,
    mixer,
    delay_value=True,
    action_space=act_spec
)
loss_module.set_keys(
    action_value=('agents','action_value'),
    local_value=('agents','chosen_action_value'),
    global_value='chosen_action_value',
    action=('agents','action')
)
loss_module.make_value_estimator(ValueEstimators.TD0, gamma=0.99)
target_updater = SoftUpdate(loss_module, eps=0.005)

# Collector & replay buffer
collector = SyncDataCollector(
    env,
    qnet_explore,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    device='cuda:0',
    storing_device='cuda:0'
)
replay_buffer = TensorDictReplayBuffer(
    storage = LazyTensorStorage(total_frames, device='cuda:0'),
    sampler = SamplerWithoutReplacement(),
    batch_size=minibatch_size
)

# Training loop
env.rollout(10,qnet)
optimizer = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)
for i, td in enumerate(collector):
    replay_buffer.extend(td.reshape(-1))
    for _ in range(num_epochs):
        batch = replay_buffer.sample()
        loss = loss_module(batch)
        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        target_updater.step()
    print(f"Iteration {i}, loss={loss['loss'].item():.4f}")
    if i >= max_iterations:
        break

collector.shutdown()
env.close()
