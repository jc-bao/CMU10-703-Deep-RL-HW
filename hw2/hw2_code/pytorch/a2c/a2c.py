import sys
import argparse
import numpy as np

import torch

# set random seeds
torch.manual_seed(0)
np.random.seed(0)

class A2C(object):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, actor, actor_lr, N, nA, critic, critic_lr, baseline=False, a2c=True):
        # Note: baseline is true if we use reinforce with baseline
        #       a2c is true if we use a2c else reinforce

        assert not (baseline and a2c), "Cannot use both baseline and a2c"
        if a2c:
            self.type = "A2C"
        elif baseline:
            self.type = "Baseline"
        else:
            self.type = "Reinforce"

        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.N = N  # N-steps
        self.nA = nA  # size of action space

    def evaluate_policy(self, env):
        # TODO: Compute Accumulative trajectory reward(set a trajectory length threshold if you want)

        rewards = self.generate_episode(env, render=False)[3]
        return np.sum(rewards)

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
   
        obss, next_obss, actions, rewards, dones = [], [], [], [], []

        obs = env.reset()
        if render:
            env.render()
        done = False

        with torch.no_grad():
            while not done:
                probs = self.actor(torch.from_numpy(obs).float())
                action = np.random.choice(self.nA, p=probs.numpy())
                next_obs, reward, done, info = env.step(action)

                if render:
                    env.render()
                
                obss.append(obs)
                next_obss.append(next_obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                obs = next_obs

        return obss, next_obss, actions, rewards, dones

    def train(self, env, gamma=0.99, n=10):
        # Trains the model on a single episode using REINFORCE or A2C/A3C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        
        obss, next_obss, actions, rewards, dones = self.generate_episode(env, render=False)

        if self.type == "Reinforce":
            T = len(obss)
            G = np.zeros(T)
            for t in range(T-1, -1, -1):
                G[t] = rewards[t] + gamma * (G[t+1] if t+1 < T else 0)
            G = torch.from_numpy(G).float()

            action_probs = self.actor(torch.from_numpy(np.array(obss)).float())[torch.arange(T), actions]

            actor_loss = -torch.sum(torch.log(action_probs) * G) / T
        else:
            raise NotImplementedError()

        # Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()