import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter


def uniform_initialize(linear_mudule, init_w):
    """
    initialize layer weight by uniform random value
    """
    linear_mudule.weight.data.uniform_(-init_w, init_w)
    linear_mudule.bias.data.uniform_(-init_w, init_w)
    return None


class LinearModule(nn.Module):
    """
    MLP module
    """

    def __init__(
        self,
        parameterspec,
        out_dim=None,
        last_activation=None,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.parameterspec = parameterspec
        self.out_dim = out_dim
        self.last_activation = last_activation
        self.activation = activation
        self.build_units()

    def build_units(self):
        self.layers = nn.ModuleList()
        for i in range(len(self.parameterspec) - 2):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.parameterspec[i], self.parameterspec[i + 1]),
                    self.activation,
                )
            )

        if self.out_dim is None:
            self.out_dim = self.parameterspec[-1]

        if self.last_activation is None:
            self.layers.append(
                nn.Sequential(nn.Linear(self.parameterspec[i + 1], self.out_dim))
            )
        else:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.parameterspec[i + 1], self.out_dim),
                    self.last_activation,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class QNetwork(nn.Module):
    """
    Q network for Critic module
    """

    def __init__(self, config, init_w=3e-3):
        super(QNetwork, self).__init__()
        default_Q_params = config.DEFAULT_MODEL_PARAMS["Q_network"]
        self.linear = LinearModule(default_Q_params, out_dim=1)
        uniform_initialize(self.linear.layers[-1].children().__next__(), init_w)

    def forward(self, states, templates, actions):
        init_tensor = torch.cat((states, templates, actions), dim=-1)
        x = self.linear(init_tensor)
        return x


class F_network(torch.nn.Module):
    """
    F network for actor module
    """

    def __init__(self, config):
        super().__init__()
        template_num = config.DEFAULT_MODEL_PARAMS["template_num"]
        default_F_params = config.DEFAULT_MODEL_PARAMS["F_network"]
        self.linear = LinearModule(default_F_params, out_dim=template_num)

    def forward(self, states):
        """
        :state {tensor}: (input_dim) bit ECFP of the current molecules
        :return {tensor} : tensor containing weights for each template (before masking)
        """
        templates = self.linear(states)
        return templates


class PolicyNetwork(nn.Module):
    """
    Policy network for actor module
    details : https://spinningup.openai.com/en/latest/algorithms/td3.html
    """

    def __init__(
        self,
        config,
        action_range=1.0,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
        device="cuda:0",
    ):
        super(PolicyNetwork, self).__init__()

        # load default parameters from config
        default_Pi_params = config.DEFAULT_MODEL_PARAMS["Pi_network"]
        num_actions = config.DEFAULT_MODEL_PARAMS["num_actions"]

        # set static parameters
        self.device = device
        self.action_range = action_range
        self.num_actions = num_actions

        # build branch linear module
        self.linear = LinearModule(default_Pi_params)

        # linear module for reparameterization
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mean_linear = nn.Linear(default_Pi_params[3], num_actions)
        self.log_std_linear = nn.Linear(default_Pi_params[3], num_actions)

        # initialize weight
        uniform_initialize(self.mean_linear, init_w)
        uniform_initialize(self.log_std_linear, init_w)

    def forward(self, states, templates_one_hot):
        init_tensor = torch.cat((states, templates_one_hot), dim=-1)
        x = self.linear(init_tensor)

        mean = torch.tanh(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        # clip the log_std into reasonable range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_noise(self, normal, noise_scale, action_shape):
        """add noise"""
        noise_clip = 2 * noise_scale
        noise = normal.sample(action_shape) * noise_scale
        noise = torch.clamp(noise, -noise_clip, noise_clip).to(self.device)
        return noise

    def evaluate(
        self,
        state,
        templates_one_hot,
        eval_noise_scale=0.1,
        deterministic=False,
        epsilon=1e-6,
    ):
        """
        predict action to learn model
        """
        mean, log_std = self.forward(state, templates_one_hot)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action_0 = 0.5 * (torch.tanh(mean + std * z.to(self.device)) + 1)
        action = (
            self.action_range * mean if deterministic else self.action_range * action_0
        )
        log_prob = (
            Normal(mean, std).log_prob(mean + std * z.to(self.device))
            - torch.log(1.0 - action_0.pow(2) + epsilon)
            - np.log(self.action_range)
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)

        noise = self.get_noise(normal, eval_noise_scale, action.shape)

        action = action + noise

        return action, log_prob, z, mean, log_std

    def get_action(
        self,
        state,
        templates_one_hot,
        deterministic=False,
        explore_noise_scale=0.1,
        without_noise=False,
    ):
        """
        generate action for interaction with env
        """
        mean, log_std = self.forward(state, templates_one_hot)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(self.device)

        action = (
            mean[0] if deterministic else (0.5 * (torch.tanh(mean + std * z) + 1))[0]
        )

        if without_noise:
            return action

        noise = self.get_noise(normal, explore_noise_scale, action.shape)
        action = self.action_range * action + noise

        return action

    def sample_action(self):
        """
        randomly sample action
        """
        # 1 x action_space
        # action in [0, 1]
        action = torch.FloatTensor(self.num_actions).uniform_(0, 1)
        return self.action_range * action


class TemplateModifier(nn.Module):
    """
    template modifier module
    apply gumbel softmax to given predicted template
    gumbel softmax patameter tau is annealed to 0.1 from 1 very slowly
    """

    def __init__(self, device):
        super().__init__()
        self.tau = Parameter(
            torch.tensor([1], dtype=torch.float32, requires_grad=False, device=device)
        )
        self.min_tau = torch.tensor(
            [0.1], dtype=torch.float32, requires_grad=False, device=device
        )
        self.n_steps = 0
        self.device = device

    def initialize(self):
        self.tau.data = torch.tensor(
            [0.1], dtype=torch.float32, requires_grad=False, device=self.device
        )
        return None

    def step(self):
        self.n_steps += 1
        # annealing 1 to 0 slowly
        lambda_ = 2e-6
        # exponentially (fast)
        # self.tau = max(0.1, self.tau * ((1-lambda_)**self.n_steps))
        # linearly (slow)
        self.tau.data = torch.max(
            self.min_tau, self.tau.data - lambda_ * (1 - 1 / self.n_steps)
        )
        return None

    def forward(self, template_mask, template):
        """
        find proper T mask for state molecule
        tau = 0 means template saturated to {0,1} hardly
        if tau is close to 1, template selection will be more stochastic
        """
        if len(template_mask.size()) == 1:
            template_mask = template_mask.unsqueeze(0).to(self.device)

        modified_template = F.gumbel_softmax(template * template_mask, tau=self.tau)
        return modified_template

    def sample_template(self, template_mask):
        """
        ramdomly sample one template
        """

        def sample_one(one_mask):
            valid_templates = torch.where(one_mask == 1)[-1]
            selected_idx = torch.randperm(len(valid_templates))[0]
            selected = valid_templates[selected_idx]
            template = torch.zeros_like(
                one_mask, dtype=torch.float32, device=self.device
            )
            template[selected] = 1
            return template

        if len(template_mask.size()) == 1:
            template_mask = template_mask.unsqueeze(0).to(self.device)

        modified_template = torch.stack(
            [sample_one(x) for x in torch.unbind(template_mask, dim=0)], dim=0
        )

        return modified_template
