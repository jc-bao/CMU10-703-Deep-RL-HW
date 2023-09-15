import numpy as np
import matplotlib.pyplot as plt
import math
import random
from dataclasses import dataclass
import tyro
from tqdm import trange
import os

random.seed(703)
np.random.seed(703)

## PROBLEM 2 : BANDITS
## In this section, we have given you a template for coding each of the
## exploration algorithms: epsilon-greedy, optimistic initialization, UCB exploration,
## and Boltzmann Exploration

## You will be implementing these algorithms as described in the “10-armed Testbed” in Sutton+Barto Section 2.3
## Please refer to the textbook or office hours if you have any confusion.

## note: you are free to change the template as you like, do not think of this as a strict guideline
## as long as the algorithm is implemented correctly and the reward plots are correct, you will receive full credit

# This is the optional wrapper for exploration algorithm we have provided to get you started
# this returns the expected rewards after running an exploration algorithm in the K-Armed Bandits problem
# we have already specified a number of parameters specific to the 10-armed testbed for guidance
# iterations is the number of times you run your algorithm

# run the experiment 1000 times!
iters = 1000

# WRAPPER FUNCTION
def explorationAlgorithmWrapper(explorationAlgorithm, param, iters):
    cumulativeRewards = []
    for i in trange(iters):
        # number of time steps
        t = 1000
        # number of arms, 10 in this instance
        k = 10
        # real reward distribution across K arms
        rewards = np.random.normal(1,1,k)
        # counts for each arm
        n = np.zeros(k)
        # extract expected rewards by running specified exploration algorithm with the parameters above
        # param is the different, specific parameter for each exploration algorithm
        # this would be epsilon for epsilon greedy, initial values for optimistic intialization, c for UCB, and temperature for Boltmann
        currentRewards = explorationAlgorithm(param, t, k, rewards, n)
        cumulativeRewards.append(currentRewards)
    # TO DO: CALCULATE AVERAGE REWARDS ACROSS EACH ITERATION TO PRODUCE EXPECTED REWARDS
    expectedRewards = np.mean(cumulativeRewards, axis=0)
    return expectedRewards

# EPSILON GREEDY TEMPLATE
def epsilonGreedy(epsilon, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    rewards = np.zeros(steps)
    # TO DO: initialize an initial q value for each arm
    Q = np.zeros(k)
    # TO DO: implement the epsilon-greedy algorithm over all steps and return the expected rewards across all steps
    for t in range(steps):
        if np.random.rand() < epsilon:
            a = np.random.choice(k)
        else:
            a = np.argmax(Q)
        reward = realRewards[a] + np.random.normal(0, 1)
        n[a] += 1
        Q[a] += (1/n[a]) * (reward - Q[a])
        rewards[t] = reward
    return rewards

# OPTIMISTIC INTIALIZATION TEMPLATE
def optimisticInitialization(value, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    rewards = np.zeros(steps)
    # TO DO: initialize optimistic initial q values per arm specified by parameter
    Q = np.full(k, value)
    # TO DO: implement the optimistic initializaiton algorithm over all steps and return the expected rewards across all steps
    for t in range(steps):
        a = np.argmax(Q)
        reward = realRewards[a] + np.random.normal(0, 1)
        n[a] += 1
        Q[a] += (1/n[a]) * (reward - Q[a])
        rewards[t] = reward
    return rewards

# UCB EXPLORATION TEMPLATE
def ucbExploration(c, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    rewards = np.zeros(steps)
    # TO DO: initialize q values per arm
    Q = np.zeros(k)
    # TO DO: implement the UCB exploration algorithm over all steps and return the expected rewards across all steps
    for t in range(steps):
        a = np.argmax(Q + c * np.sqrt(np.log(t+1) / (n+1)))
        reward = realRewards[a] + np.random.normal(0, 1)
        n[a] += 1
        Q[a] += (1/n[a]) * (reward - Q[a])
        rewards[t] = reward
    return rewards


# BOLTZMANN EXPLORATION TEMPLATE
def boltzmannE(temperature, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    rewards = np.zeros(steps)
    # TO DO: initialize q values per arm
    Q = np.zeros(k)
    # TO DO: initialize probability values for each arm
    prob = np.exp(Q * temperature) / np.sum(np.exp(Q * temperature))
    # TO DO: implement the Boltzmann Exploration algorithm over all steps and return the expected rewards across all steps
    for t in range(steps):
        a = np.random.choice(k, p=prob)
        reward = realRewards[a] + np.random.normal(0, 1)
        n[a] += 1
        Q[a] += (1/n[a]) * (reward - Q[a])
        rewards[t] = reward
        prob = np.exp(Q * temperature) / np.sum(np.exp(Q * temperature))
    return rewards

# PLOT TEMPLATE
def plotExplorations(paramList, explorationAlgorithmWrapper, title, sample_fn):
    # TO DO: for each parameter in the param list, plot the returns from the exploration Algorithm from each param on the same plot
    # Zhicheng: What's this?
    x = np.arange(1,1001)
    plt.figure(figsize=(12,8))
    for param in paramList:
        # calculate your Ys (expected rewards) per each parameter value
        y = explorationAlgorithmWrapper(sample_fn, param, iters)
        # plot all the Ys on the same plot
        plt.plot(x, y, label=f'param={param}')
        # save the data
        if not os.path.exists('data'):
            os.makedirs('data')
        np.save(f'data/bandit_{title}_param_{param}.npy', y)
    plt.xlabel('Steps')
    plt.ylabel('Expected Rewards')
    plt.title(title)
    # include correct labels on your plot!
    plt.legend()
    # save plot
    plt.savefig(f'bandit_{title}.png')

def plotBest():
    # get all files's name from 'data/best' folder
    files = os.listdir('data/best')
    # parse all files name in format bandit_{title}_param_{param}.npy, get title and param
    titles = []
    params = []
    for file in files:
        title, param = file.split('_param_')
        param = param[:-4]
        titles.append(title[7:])
        params.append(param)
    # plot all files
    x = np.arange(1,1001)
    plt.figure(figsize=(12,8))
    for i in range(len(files)):
        file = files[i]
        y = np.load(f'data/best/{file}')
        plt.plot(x, y, label=f"{titles[i]}-{params[i]}")
    plt.xlabel('Steps')
    plt.ylabel('Expected Rewards')
    plt.title('Best performance comparison')
    plt.legend()
    plt.savefig(f'bandit_best.png')

@dataclass
class Args:
    mode: str = "greedy"

def main(args: Args):
    if args.mode == 'greedy':
        plotExplorations([0, 0.001, 0.01, 0.1, 1.0], explorationAlgorithmWrapper, 'Epsilon Greedy', epsilonGreedy)
    elif args.mode == 'optimistic':
        plotExplorations([0, 1, 2, 5, 10], explorationAlgorithmWrapper, 'Optimistic Initialization', optimisticInitialization)
    elif args.mode == 'ucb':
        plotExplorations([0, 1, 2, 5], explorationAlgorithmWrapper, 'UCB Exploration', ucbExploration)
    elif args.mode == 'boltzmann':
        plotExplorations([1, 3, 10, 30, 100], explorationAlgorithmWrapper, 'Boltzmann Exploration', boltzmannE)
    elif args.mode == 'plot_best':
        plotBest()
    else:
        raise ValueError(f"Unknown mode {args.mode}")

if __name__ == "__main__":
    main(tyro.cli(Args))