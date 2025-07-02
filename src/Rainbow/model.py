from __future__ import division
import math
from functools import partial
from typing import Sequence

import torch
import torchrl
from torch import nn
from torch.nn import functional as F, Sequential, Flatten
import numpy as np
from torch import nn
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import MultiAgentNetBase, NoisyLazyLinear
from torchrl.modules.models import ConvNet, MLP
from torchrl.modules.models.utils import _reset_parameters_recursive



# Factorised NoisyLinear layer with bias

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5,bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Register static buffers — do not dynamically assign
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()

    @property
    def is_meta(self):
        return self.weight_mu.device.type == "meta"

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight_mu.uniform_(-mu_range, mu_range)
            self.weight_sigma.fill_(self.std_init / math.sqrt(self.in_features))
            self.bias_mu.uniform_(-mu_range, mu_range)
            self.bias_sigma.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        return torch.randn(size, device=self.weight_mu.device).sign().mul_(
            torch.randn(size, device=self.weight_mu.device).abs().sqrt_()
        )

    def reset_noise(self):
        if self.is_meta:
            return
        with torch.no_grad():
            eps_in = self._scale_noise(self.in_features)
            eps_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(torch.ger(eps_out, eps_in))
            self.bias_epsilon.copy_(eps_out)

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.convs = nn.Sequential(nn.Conv2d(12, 32, 3, stride=1, padding='same'), nn.ReLU(),
                                       nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),  # stride 1
                                       nn.Conv2d(64, 64, 3, stride=2),
                                       nn.ReLU())  # kernel size 3 , stride 2, stable version had kernel 4
        self.conv_output_size = self._get_conv_out([12, 41, 41])
        self.fc_1_v = NoisyLinear(self.conv_output_size + 40, args.hidden_size, std_init=args.noisy_std)
        self.fc_1_a = NoisyLinear(self.conv_output_size + 40, args.hidden_size, std_init=args.noisy_std)
        self.fc_2_v = NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_2_a = NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_3_v = NoisyLinear(args.hidden_size, 1, std_init=args.noisy_std)
        self.fc_3_a = NoisyLinear(args.hidden_size, action_space, std_init=args.noisy_std)

    def _get_conv_out(self, shape):
        o = self.convs(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, b, a, o):
        b = b.view(-1, 1)
        a = a.view(a.size(0), -1)
        o = o.view(o.size(0), -1)
        x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, b), 1)
        x = torch.cat((x, a), 1)
        x = torch.cat((x, o), 1)
        v = self.fc_3_v(F.relu(self.fc_2_v(F.relu(self.fc_1_v(x))))) # Value stream
        a = self.fc_3_a(F.relu(self.fc_2_a(F.relu(self.fc_1_a(x)))))  # Advantage stream
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()


class QNetwork(nn.Module):
    def __init__(self,device,**kwargs):
        self.device = device
        super(QNetwork, self).__init__()
        self.action_space = 5
        self.flatten_input = Flatten(start_dim=-4, end_dim=-3)
        self.flatten_o = Flatten(start_dim=-3,end_dim=-1)
        self.flatten_a = Flatten(start_dim=-2,end_dim=-1)
        self.convs =  ConvNet(
            in_features=12,
            num_cells=[32, 64, 64],
            kernel_sizes=[3, 3, 3],
            strides=[1, 2, 2],
            paddings=[1, 1, 1],
            activation_class=torch.nn.ReLU,
            device=device,
            **kwargs,
        )

        #self.fc_1_v = torchrl.modules.NoisyLinear(self.conv_output_size, 512, std_init=0.5,device = device)
        #self.fc_1_a = torchrl.modules.NoisyLinear(self.conv_output_size, 512, std_init=0.5,device = device)
        #self.fc_2_v = torchrl.modules.NoisyLinear(512, 512, std_init=0.5,device = device)
        #self.fc_2_a = torchrl.modules.NoisyLinear(512, 512, std_init=0.5,device = device)
        #self.fc_3_v = torchrl.modules.NoisyLinear(512, 1, std_init=0.5,device = device)
        #self.fc_3_a = torchrl.modules.NoisyLinear(512, self.action_space, std_init=0.5,device = device)
        self.value_head = MLP(in_features=7744+40,
                  out_features=1,
                  depth=3,
                  num_cells=512,
                  device=device,
                  activation_class=torch.nn.ReLU,
         #         layer_class=torchrl.modules.NoisyLinear
                                      )# fix the bias value here
        #       #self.value_head= Sequential(self.convs,self.value_head)
        self.advantage_head = MLP(in_features=7744+40,
                   out_features=5,
                   depth=3,
                   num_cells=512,
                   device=device,
                   activation_class=torch.nn.ReLU,)
        #           layer_class=torchrl.modules.NoisyLinear   )
        #self.advantage_head = Sequential(self.convs, self.advantage_head)


    def _get_conv_out(self, shape):
        o = self.convs(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        s,b,a,o = x
        s = self.flatten_input(s)
        x = self.convs(s)
        x = torch.cat((x, b), -1)
        a= self.flatten_a(a)
        x = torch.cat((x, a), -1)
        o = self.flatten_o(o)
        x = torch.cat((x, o), -1)
        v = self.value_head(x)
        a = self.advantage_head(x)
        q = v + a - a.mean(-1, keepdim=True)  # Combine streams
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'head' in name:
                module.reset_noise()




class MyNet(MultiAgentNetBase):

    def __init__(
        self,
        n_agents: int,
        centralized: bool | None = None,
        share_params: bool | None = None,
        *,
        device: DEVICE_TYPING | None = None,
        use_td_params: bool = True,
        **kwargs,
    ):

        super().__init__(
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            device=device,
            agent_dim=-5,
            use_td_params=use_td_params,
            **kwargs,
        )

    def _build_single_net(self,device,**kwargs):


        return QNetwork(device)

    def _pre_forward_check(self, inputs):
        return inputs
        if len(inputs.shape) < 4:
            raise ValueError(
                """Multi-agent network expects (*batch_size, agent_index, x, y, channels)"""
            )
        if inputs.shape[-5] != self.n_agents:
            raise ValueError(
                f"""Multi-agent network expects {self.n_agents} but got {inputs.shape[-4]}"""
            )
        if self.centralized:
            raise ValueError(
                f"""NOT IMPLEMENTED"""
            )
        return inputs

    def forward(self, *inputs: tuple[torch.Tensor]) -> torch.Tensor:
        inputs = self._pre_forward_check(inputs)
        # If parameters are not shared, each agent has its own network
        if not self.share_params:
            if self.centralized:
                output = self.vmap_func_module(
                    self._empty_net, (0, None), (-2,), randomness=self.vmap_randomness
                )(self.params, inputs)
            else:
                output = self.vmap_func_module(
                    self._empty_net,
                    (0, self.agent_dim),
                    (-2,),
                    randomness=self.vmap_randomness,
                )(self.params, inputs)

        # If parameters are shared, agents use the same network
        else:
            with self.params.to_module(self._empty_net):
                output = self._empty_net(inputs)

            if self.centralized:
                # If the parameters are shared, and it is centralized, all agents will have the same output
                # We expand it to maintain the agent dimension, but values will be the same for all agents
                n_agent_outputs = output.shape[-1]
                output = output.view(*output.shape[:-1], n_agent_outputs)
                output = output.unsqueeze(-2)
                output = output.expand(
                    *output.shape[:-2], self.n_agents, n_agent_outputs
                )

        if output.shape[-2] != (self.n_agents):
            raise ValueError(
                f"Multi-agent network expected output with shape[-2]={self.n_agents}"
                f" but got {output.shape}"
            )

        return output