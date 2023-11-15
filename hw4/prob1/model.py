import os
import numpy as np
import torch
import torch.nn as nn
import operator
from functools import reduce
from utils.util import ZFilter
from tqdm import tqdm

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

import logging
log = logging.getLogger('root')


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        self.max_logvar = torch.tensor(-3 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)
        self.min_logvar = torch.tensor(-7 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)

        # Create or load networks
        self.networks = nn.ModuleList([self.create_network(n) for n in range(self.num_nets)]).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

    def get_output(self, output):
        """
        Argument:
          output: the raw output of a single ensemble member
        Return:
          mean and log variance
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_loss(self, targ, mean, logvar):
        '''
        targ: (batch_size, state_dim)
        mean: (batch_size, state_dim)
        logvar: (batch_size, state_dim)
        '''
        loss = torch.nn.GaussianNLLLoss()
        return loss(mean, targ, logvar.exp())

    def create_network(self, n):
        layer_sizes = [self.state_dim + self.action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, HIDDEN3_UNITS]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)
    
    def train_step(self, batch_inputs, batch_targets):
        self.opt.zero_grad()
        batch_inputs = batch_inputs.to(self.device)
        batch_targets = batch_targets.to(self.device)
        batch_outputs = self.forward(batch_inputs)
        total_loss = 0
        for i in range(self.num_nets):
            loss = self.get_loss(batch_targets[i], batch_outputs[i][0], batch_outputs[i][1])
            total_loss += loss
        total_loss.backward()
        self.opt.step()
        return total_loss / self.num_nets

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
        Argument:
          inputs: state and action inputs. Assumes that inputs are standardized.
          targets: resulting states
        Return:
            List containing the average loss of all the networks at each train iteration

        """
        # TODO: revisit https://piazza.com/class/llmlh1ivh0156m/post/279
        losses = []
        for epoch in tqdm(range(num_train_itrs), desc="Training"):
            permutation = torch.randperm(inputs.shape[0])
            avg_loss = 0
            for start in range(0, inputs.shape[0], batch_size):
                end = min(start + batch_size, inputs.shape[0])
                indices = permutation[start:end]
                batch_inputs, batch_targets = inputs[indices], targets[indices]
                avg_loss += self.train_step(batch_inputs, batch_targets) * (end - start)
            avg_loss = avg_loss / inputs.shape[0]
            losses.append(avg_loss.item())

        return losses