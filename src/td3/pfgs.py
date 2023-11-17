"""
implementation of PGFS model from https://arxiv.org/abs/2004.12485
"""
import random
from collections import deque

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn

from td3.actor_critic import Actor, Critic


def convert_SMILES_to_FP(smiles: str, bits: int) -> np.array:
    mol = Chem.MolFromSmiles(smiles)
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
    fp = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(morgan, fp)
    return fp


def smiles_to_FP_tensor(smiles: str, device=None, bits=1024):
    state_fp = convert_SMILES_to_FP(smiles, bits)
    if device:
        state_fp = torch.FloatTensor(state_fp).unsqueeze(0).to(device)
    else:
        state_fp = torch.FloatTensor(state_fp).unsqueeze(0)
    return state_fp


class PGFS(nn.Module):
    """
    PGFS module based on TD3 algorithm
    """

    def __init__(self, config, device):
        super().__init__()
        self.actor = Actor(config, device)
        self.critic = Critic(config, device)
        self.config = config
        self.replay_buffer = deque(maxlen=config.DEFAULT_LEARNING_PARAMS["max_memory"])

        self.batch_size = config.DEFAULT_LEARNING_PARAMS["batch_size"]
        # discount factor for Bellman eq.
        self.gamma = config.DEFAULT_LEARNING_PARAMS["gamma"]
        self.device = device
        self.interval_checker = 0
        # delayed update interval
        self.policy_target_update_interval = config.DEFAULT_LEARNING_PARAMS[
            "policy_target_update_interval"
        ]

    def memorize(
        self,
        state_smiles,
        next_states_smiles,
        template_mask,
        next_template_mask,
        template,
        action,
        reward,
        done,
    ):
        """
        memorize the current step with actions and rewards
        """
        self.replay_buffer.append(
            tuple(
                [
                    state_smiles,
                    next_states_smiles,
                    template_mask.unsqueeze(0),
                    next_template_mask.unsqueeze(0),
                    template,
                    action.unsqueeze(0),
                    reward,
                    done.unsqueeze(0),
                ]
            )
        )

    def load_batch_from_buffer(self):
        """
        load training features from replay buffer and assignment it to device
        """
        # unpacking batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        (
            state_smiles,
            next_states_smiles,
            template_mask,
            next_template_mask,
            templates,
            actions,
            rewards,
            dones,
        ) = zip(*batch)

        # convert to FP
        states_fp, next_states_fp = [
            [
                smiles_to_FP_tensor(smiles, device=self.device, bits=1024)
                for smiles in state
            ]
            for state in [state_smiles, next_states_smiles]
        ]

        # assign to device
        def assignment_factory(feature, device):
            return torch.cat(feature).to(device)

        batch_assigned = [
            assignment_factory(feature, self.device)
            for feature in [
                states_fp,
                next_states_fp,
                templates,
                template_mask,
                next_template_mask,
                actions,
                dones,
            ]
        ]
        # reward scaling (generally gives good effects on learning)
        rewards = torch.cat(rewards).unsqueeze(-1).to(self.device)
        reward_scale = 1.0
        rewards = (
            reward_scale * (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-6)
        )

        return batch_assigned, rewards

    def calculate_actor_loss(self, template_mask, states_fp):
        (
            action,
            template_modified,
            template_pred,
            template_label,
            _,
            _,
            _,
            _,
        ) = self.actor.predict_action(template_mask, states_fp, return_mask=True)
        criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
        f_loss = criterion(template_pred, template_label) / self.batch_size

        q_val = self.critic.predict_q_value(states_fp, template_modified, action)
        # use just one q_value
        pi_loss = -q_val[0].sum()
        return f_loss, pi_loss, q_val

    def calculate_critic_loss(self, inputs):
        (
            states_fp,
            templates,
            actions,
            next_template_mask,
            next_states_fp,
            rewards,
            dones,
        ) = inputs
        current_qs = self.critic.predict_q_value(states_fp, templates, actions)

        with torch.no_grad():
            next_actions, next_templates, _, _, _, _ = self.actor.predict_action(
                next_template_mask, next_states_fp, use_target_net=True
            )

            target_min_q = self.critic.predict_q_value(
                next_states_fp, next_templates, next_actions, use_target_net=True
            )
        target_q = rewards + (1 - dones) * self.gamma * target_min_q

        q_loss = [
            ((current_q - target_q.detach()) ** 2).mean() for current_q in current_qs
        ]

        return q_loss

    def update(self):
        """
        calculate loss by samples from replay buffer
        then update network by using calculated loss
        """
        if self.replay_buffer.__len__() < self.batch_size:
            return

        self.interval_checker += 1
        # load features from buffer
        (
            states_fp,
            next_states_fp,
            templates,
            template_mask,
            next_template_mask,
            actions,
            dones,
        ), rewards = self.load_batch_from_buffer()

        # critic network update
        critic_featurs = (
            states_fp,
            templates,
            actions,
            next_template_mask,
            next_states_fp,
            rewards,
            dones,
        )
        q_loss = self.calculate_critic_loss(critic_featurs)
        self.critic.update_gradient(q_loss)

        # step for tau annealings
        self.actor.template_modifier.step()

        f_loss, pi_loss, q_val = self.calculate_actor_loss(template_mask, states_fp)

        # update actor network intervally
        if self.interval_checker == self.policy_target_update_interval:
            self.actor.update_gradient(f_loss, pi_loss)
            # soft update for target network
            self.critic.soft_update()
            self.actor.soft_update()

            # reset interval checker
            self.interval_checker = 0

        return (
            (q_loss[0].item() + q_loss[1].item()) / 2,
            f_loss.item(),
            pi_loss.item(),
            q_val[0].mean().item(),
        )

    def predict(self, template_mask, state_fp, training=True):
        """
        predict action by actor network
        """
        # without noise
        action, moditied_template = self.actor.get_action(
            template_mask, state_fp, without_noise=not (training)
        )
        return action, moditied_template

    def sample(self, template_mask):
        """
        randomly sample action
        """
        action, moditied_template = self.actor.random_search(template_mask)
        return action.to(self.device), moditied_template.to(self.device)
