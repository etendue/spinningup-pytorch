import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
#from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete


LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-6

def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh, output_activation=None,
                 output_scale=1, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return torch.squeeze(x) if self.output_squeeze else x


class CategoricalPolicy(nn.Module):
    def __init__(self, in_features, hidden_sizes,
                 activation, output_activation,
                 action_space):
        super(CategoricalPolicy, self).__init__()

        action_dim = action_space.n #discrete action space
        # self.action_scale = torch.Tensor(action_space.high[0])
        self.output_activation = output_activation

        self.net = MLP(layers=[in_features]+list(hidden_sizes),
                      activation=activation,
                      output_activation=activation)

        self.logits = nn.Linear(in_features=list(hidden_sizes)[-1],
                            out_features=action_dim)

    def forward(self, x):
        output = self.net(x)
        logits = self.logits(output)
        return logits


class ActorCritic(nn.Module):
    def __init__(self, in_features, action_space,
                 hidden_sizes=(400, 300), activation=torch.relu,
                 output_activation=None, policy=CategoricalPolicy):
        super(ActorCritic, self).__init__()

        action_dim = action_space.n # discrete action space

        self.policy = policy(in_features, hidden_sizes, activation,
                             output_activation, action_space)

        self.vf_mlp = MLP([in_features]+list(hidden_sizes)+[1],
                          activation, output_squeeze=True)

        self.q1 = MLP([in_features]+list(hidden_sizes)+[action_dim],
                      activation, output_squeeze=True)

        self.q2 = MLP([in_features]+list(hidden_sizes)+[action_dim],
                      activation, output_squeeze=True)

    def forward(self, x, a):
        logits = self.policy(x)

        m = Categorical(logits=logits)
        pi = m.sample()

        logp_pi = m.log_prob(pi)
        logp_a = m.log_prob(a)
        entropy = m.entropy()
        q1_all = self.q1(x)
        q2_all = self.q2(x)

        q1 = q1_all.gather(dim=1, index=a.view(-1, 1)).squeeze()
        q2 = q2_all.gather(dim=1, index=a.view(-1, 1)).squeeze()
        q1_pi = q1_all.gather(dim=1, index=pi.view(-1, 1)).squeeze()
        q2_pi = q2_all.gather(dim=1, index=pi.view(-1, 1)).squeeze()
        v = self.vf_mlp(x)

        return q1, q2, logp_a, q1_pi, q2_pi, logp_pi, entropy,v
