# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class CNN_ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        imgsz = (640,480),
        channels = 3,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs - imgsz[0] * imgsz[1] * channels
        mlp_input_dim_c = num_critic_obs - imgsz[0] * imgsz[1] * channels

        self.mlp_input_dim_a = mlp_input_dim_a
        # print(self.mlp_input_dim_a)
        self.mlp_input_dim_c = mlp_input_dim_c
        # Policy
        self.actor_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            activation,
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            activation,
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=0),
            activation,
            nn.Flatten(),
            nn.Linear(32 * 1064 , 512),  # Adjust dimensions based on input size and convolution layers
            activation,
        )

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index != len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor_mlp = nn.Sequential(*actor_layers)

        self.actor = nn.Sequential(nn.Linear(actor_hidden_dims[-1] + 512, num_actions), activation)
        # Value function

        self.critic_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            activation,
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            activation,
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=0),
            activation,
            nn.Flatten(),
            nn.Linear(32 * 1064 , 512),  # Adjust dimensions based on input size and convolution layers
            activation,
        )

        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index != len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic_mlp = nn.Sequential(*critic_layers)

        self.critic = nn.Sequential(nn.Linear(critic_hidden_dims[-1] + 512, 1), activation)

        print(f"Actor MLP: {self.actor_mlp}")
        print(f"actConv layers: {self.actor_conv}")
        print(f"Critic MLP: {self.critic}")
        print(f"criticConv layers: {self.critic_conv}")
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mlp_in = observations[:, :self.mlp_input_dim_a]
        conv_in = observations[:, self.mlp_input_dim_a:].view(-1, 3, 640, 480)
        mlp_out = self.actor_mlp(mlp_in)
        conv_out = self.actor_conv(conv_in)
        actor_in = torch.cat((mlp_out, conv_out), dim=1)
        mean = self.actor(actor_in)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        mlp_in = observations[:, :self.mlp_input_dim_a]
        conv_in = observations[:, self.mlp_input_dim_a:].view(-1, 3, 640, 480)
        mlp_out = self.actor_mlp(mlp_in)
        conv_out = self.actor_conv(conv_in)
        actor_in = torch.cat((mlp_out, conv_out), dim=1)
        actions_mean = self.actor(actor_in)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        mlp_in = critic_observations[:, :self.mlp_input_dim_a]
        conv_in = critic_observations[:, self.mlp_input_dim_a:].view(-1, 3, 640, 480)
        mlp_out = self.critic_mlp(mlp_in)
        conv_out = self.critic_conv(conv_in)
        critic_in = torch.cat((mlp_out, conv_out), dim=1)
        value = self.critic(critic_in)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True

if __name__ == "__main__":
    # Example usage
    test_input = torch.zeros((1,921835))
    model = CNN_ActorCritic(num_actor_obs=921835, num_critic_obs=921835, num_actions=12, imgsz=(640, 480), channels=3)
    # print(model)
    test_output = model.act(test_input)
    print(test_output.shape)
