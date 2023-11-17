"""
actor critic module 
based on TD3 algorithm
about TD3 : https://spinningup.openai.com/en/latest/algorithms/td3.html
"""
import torch
import torch.nn as nn
import copy
from td3.networks import QNetwork, PolicyNetwork, F_network, TemplateModifier


# calculate loss slight differently cause torch bug
# torch v1.4 gradient calculation error issue
# https://discuss.pytorch.org/t/solved-pytorch1-5-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/90256/11
TORCH_VERSION = float(torch.__version__[:3])


def target_soft_update(net, target_net, soft_tau=0.005):
    """
    soft update for target network
    W_target_net = tau * (W_target_net) + (1-tau) * W_net
    """
    # Soft update the target net
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(  # copy data value into target parameters
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    return target_net


class Actor(nn.Module):
    """
    actor moodule for actor-critic method
    pi_network : determine action for given (s_t, template_t)
    pi_target_network : determine future action for given (s_t+1, template_t+1)
    f_network : predict best template for given (s_t)
    template_modifier : from predicted template, generate modified template by applying gumbel softmax
    """

    def __init__(self, config, device):
        """
        use default structure
        features needs to be normalized
        """
        super().__init__()
        self.pi_network = PolicyNetwork(config=config, device=device).to(device)
        self.pi_target_network = copy.deepcopy(self.pi_network).to(device)
        self.f_network = F_network(config).to(device)
        self.template_modifier = TemplateModifier(device).to(device)
        self.lr_f, self.lr_pi = (
            config.DEFAULT_LEARNING_PARAMS["lr_f"],
            config.DEFAULT_LEARNING_PARAMS["lr_pi"],
        )
        self.set_optimaizer()

    def set_optimaizer(self):
        self.f_optimizer = torch.optim.Adam(self.f_network.parameters(), self.lr_f)

        self.pi_optimizer = torch.optim.Adam(
            list(self.pi_network.parameters()) + list(self.f_network.parameters()),
            self.lr_pi,
        )

    def predict_action(
        self, template_mask, states, use_target_net=False, return_mask=False
    ):
        """
        predict action to learn model
        """
        template = self.f_network(states)
        moditied_template = self.template_modifier(template_mask, template)
        network = self.pi_network
        if use_target_net:
            network = self.pi_target_network

        action, log_prob, z, mean, log_std = network.evaluate(states, moditied_template)

        if return_mask:
            return (
                action,
                moditied_template,
                template,
                template_mask,
                log_prob,
                z,
                mean,
                log_std,
            )

        return action, moditied_template, log_prob, z, mean, log_std

    def get_action(self, template_mask, states, without_noise=False):
        """
        get action for generate sample
        """
        template = self.f_network(states)
        moditied_template = self.template_modifier(template_mask, template)
        action = self.pi_network.get_action(
            states, moditied_template, without_noise=without_noise
        )
        return action, moditied_template

    def random_search(self, template_mask):
        """
        random search for initial learning stage
        """
        moditied_template = self.template_modifier.sample_template(template_mask)
        action = self.pi_network.sample_action()
        return action, moditied_template

    def update_gradient(self, f_loss, pi_loss):
        """
        update networks by calculated loss
        """
        self.f_optimizer.zero_grad()
        self.pi_optimizer.zero_grad()

        # version check
        if TORCH_VERSION == 1.4:
            f_loss.backward(retain_graph=True)
            pi_loss.backward()
        else:
            f_loss.backward(inputs=list(self.f_network.parameters()), retain_graph=True)
            pi_loss.backward(
                inputs=list(self.pi_network.parameters())
                + list(self.f_network.parameters())
            )

        self.f_optimizer.step()
        self.pi_optimizer.step()

        return None

    def soft_update(self):
        """
        soft update for target networks
        """
        self.pi_target_network = target_soft_update(
            self.pi_network, self.pi_target_network
        )
        return None


class Critic(nn.Module):
    def __init__(self, config, device):
        """
        critic moodule for actor-critic method
        clipped Q learning (to prevent to overestimate Q) is applied.
        that means, it has two pairs of networks for each Q and Q_target
        """
        super().__init__()
        self.config = config
        self.lr_q = config.DEFAULT_LEARNING_PARAMS["lr_q"]
        self.device = device
        self.build_networks()
        self.build_optimaizer()

    def build_networks(self):
        self.q_network_1 = QNetwork(self.config).to(self.device)
        self.q_network_2 = QNetwork(self.config).to(self.device)
        self.q_target_network_1 = copy.deepcopy(self.q_network_1).to(self.device)
        self.q_target_network_2 = copy.deepcopy(self.q_network_2).to(self.device)

    def build_optimaizer(self):
        self.q_optimizer = torch.optim.Adam(self.q_network_1.parameters(), self.lr_q)

        self.q_optimizer_2 = torch.optim.Adam(self.q_network_2.parameters(), self.lr_q)

    def update_gradient(self, loss):
        """
        update networks by calculated loss
        """
        self.q_optimizer.zero_grad()
        self.q_optimizer_2.zero_grad()

        # version check
        if TORCH_VERSION == 1.4:
            loss[0].backward(retain_graph=True)
            loss[1].backward(retain_graph=True)
        else:
            loss[0].backward(
                inputs=list(self.q_network_1.parameters()), retain_graph=True
            )
            loss[1].backward(
                inputs=list(self.q_network_2.parameters()), retain_graph=True
            )

        self.q_optimizer.step()
        self.q_optimizer_2.step()

        return None

    def predict_q_value(self, states, templates, actions, use_target_net=False):
        """
        predict Q value for (s_t, template_t)
        by clipped Q method
        """
        network = [self.q_network_1, self.q_network_2]
        if use_target_net:
            network = [self.q_target_network_1, self.q_target_network_2]

        q_values = [q_net(states, templates, actions) for q_net in network]
        min_q_value = torch.min(*q_values)
        if not (use_target_net):
            return q_values
        return min_q_value

    def soft_update(self):
        """
        soft update for target networks
        """
        self.q_target_network_1 = target_soft_update(
            self.q_network_1, self.q_target_network_1
        )

        self.q_target_network_2 = target_soft_update(
            self.q_network_2, self.q_target_network_2
        )
        return None
