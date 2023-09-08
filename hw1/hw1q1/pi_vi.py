# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gym
import lake_envs as lake_env
from dataclasses import dataclass
import tyro


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have observation_space, action_space, and
      P as attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    policy = np.zeros(env.observation_space.n, dtype=np.int)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum(p*(r + gamma*value_function[s_]) for p, s_, r, _ in env.P[s][a])
        policy[s] = np.argmax(q_sa)
    return policy

def evaluate_policy_sync(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    num_iter = 0

    while num_iter < max_iterations:
        delta = 0
        value_func_new = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            value_func_new[s] = sum(
                prob * (rew + gamma * value_func[s_])
                for prob, s_, rew, _ in env.P[s][policy[s]]
            )
            delta = max(delta, abs(value_func_new[s] - value_func[s]))
        value_func = value_func_new
        num_iter += 1
        if delta < tol:
            break
    return value_func_new, num_iter


def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    num_iter = 0
    while num_iter<max_iterations:
        delta = 0
        for s in range(env.observation_space.n):
            v = value_func[s]
            value_func[s] = sum(
                prob * (rew + gamma * value_func[s_])
                for prob, s_, rew, _ in env.P[s][policy[s]]
            )
            delta = max(delta, abs(v - value_func[s]))
        num_iter += 1
        if delta < tol:
            return value_func, num_iter
    return value_func, num_iter


def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    num_iter = 0
    while num_iter<max_iterations:
        delta = 0
        states = np.random.permutation(env.observation_space.n)
        for s in states:
            v = value_func[s]
            value_func[s] = sum(
                prob * (rew + gamma * value_func[s_])
                for prob, s_, rew, _ in env.P[s][policy[s]]
            )
            delta = max(delta, abs(v - value_func[s]))
        num_iter += 1
        if delta < tol:
            return value_func, num_iter
    return value_func, num_iter

def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True
    for s in range(env.observation_space.n):
        old_action = policy[s]
        policy[s] = np.argmax(
            [
                sum(
                    prob * (rew + gamma * value_func[s_])
                    for prob, s_, rew, _ in env.P[s][a]
                )
                for a in range(env.action_space.n)
            ]
        )
        if old_action != policy[s]: 
            policy_stable = False
    return not policy_stable, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n)) 
    value_func = np.zeros(env.observation_space.n)
    num_policy_iter = 0
    num_value_iter = 0

    while num_policy_iter < max_iterations:
        # Policy Evaluation
        value_func, iterations = evaluate_policy_sync(env, value_func, gamma, policy, max_iterations, tol)
        num_value_iter += iterations

        # Policy Improvement
        policy_changed, policy = improve_policy(env, gamma, value_func, policy)
        num_policy_iter += 1

        if not policy_changed:
            break

    return policy, value_func, num_policy_iter, num_value_iter

def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    value_func = np.zeros(env.observation_space.n)
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))
    num_policy_iter = 0
    num_value_iter = 0

    while num_policy_iter < max_iterations:
        value_func, num_value_iter_current = evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iterations, tol)
        num_value_iter += num_value_iter_current

        # Policy Improvement
        policy_changed, policy = improve_policy(env, gamma, value_func, policy)
        num_policy_iter += 1

        if not policy_changed:
            break

    return policy, value_func, num_policy_iter, num_value_iter


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    value_func = np.zeros(env.observation_space.n)
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))
    num_policy_iter = 0
    num_value_iter = 0

    while num_policy_iter < max_iterations:
        value_func, num_value_iter_current = evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iterations, tol)
        num_value_iter += num_value_iter_current

        # Policy Improvement
        policy_changed, policy = improve_policy(env, gamma, value_func, policy)
        num_policy_iter += 1

        if not policy_changed:
            break

    return policy, value_func, num_policy_iter, num_value_iter

def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    num_iter = 0
    value_func = np.zeros(env.observation_space.n)
    while num_iter < max_iterations:
        delta = 0
        value_func_new = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            value_func_new[s] = np.max(
                [
                    sum(p * (r + gamma * value_func[s_]) for p, s_, r, _ in env.P[s][a])
                    for a in range(env.action_space.n)
                ]
            )
            delta = max(delta, abs(value_func_new[s] - value_func[s]))
            value_func[s] = value_func_new[s]
        value_func = value_func_new
        num_iter += 1
        if delta < tol:
            return value_func, num_iter
    return value_func, num_iter

def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    num_iter = 0
    value_func = np.zeros(env.observation_space.n)
    while num_iter < max_iterations:
        delta = 0
        for s in range(env.observation_space.n):
            v = value_func[s]
            value_func[s] = np.max(
                [
                    sum(p * (r + gamma * value_func[s_]) for p, s_, r, _ in env.P[s][a])
                    for a in range(env.action_space.n)
                ]
            )
            delta = max(delta, abs(value_func[s] - v))
        num_iter += 1
        if delta < tol:
            return value_func, num_iter
    return value_func, num_iter



def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    num_iter = 0
    value_func = np.zeros(env.observation_space.n)
    while num_iter < max_iterations:
        delta = 0
        states = np.random.permutation(env.observation_space.n)
        for s in states:
            v = value_func[s]
            value_func[s] = np.max(
                [
                    sum(p * (r + gamma * value_func[s_]) for p, s_, r, _ in env.P[s][a])
                    for a in range(env.action_space.n)
                ]
            )
            delta = max(delta, abs(value_func[s] - v))
        num_iter += 1
        if delta < tol:
            return value_func, num_iter
    return value_func, num_iter

def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have observation_space,
      action_space, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    world_size = int(env.observation_space.n ** 0.5)
    assert world_size ** 2 == env.observation_space.n, "world size must be a perfect square"
    goal=57
    dist_fn = lambda s: abs(s % world_size - goal % world_size) + abs(s // world_size - goal // world_size)
    states_ordered = sorted(range(env.observation_space.n), key=dist_fn)

    num_iter = 0
    value_func = np.zeros(env.observation_space.n)
    while num_iter < max_iterations:
        delta = 0
        for s in states_ordered:
            v = value_func[s]
            value_func[s] = np.max(
                [
                    sum(p * (r + gamma * value_func[s_]) for p, s_, r, _ in env.P[s][a])
                    for a in range(env.action_space.n)
                ]
            )
            delta = max(delta, abs(value_func[s] - v))
        num_iter += 1
        if delta < tol:
            return value_func, num_iter
    return value_func, num_iter


######################
#  Optional Helpers  #
######################

# Here we provide some helper functions simply for your convinience.
# You DON'T necessarily need them, especially "env_wrapper" if
# you want to deal with it in your different ways.

# Feel FREE to change/delete these helper functions.

def display_policy_letters(env, policy):
    """Displays a policy as letters, as required by problem 1.2 & 1.3

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.observation_space.n)
    """
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_env.action_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)


    for row in range(env.nrow):
        print(''.join(policy_letters[row, :]))


def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)

    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros(
    (env.observation_space.n, env.action_space.n, env.observation_space.n))
    env.R = np.zeros(
    (env.observation_space.n, env.action_space.n, env.observation_space.n))

    for state in range(env.observation_space.n):
      for action in range(env.action_space.n):
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
            env.T[state, action, nextstate] = prob
            env.R[state, action, nextstate] = reward
    return env


def value_func_heatmap(env, value_func):
    """Visualize a policy as a heatmap, as required by problem 1.2 & 1.3

    Note that you might need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.observation_space.n)
    """
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]),
                annot=False, linewidths=.5, cmap="GnBu_r", ax=ax,
                yticklabels = np.arange(1, env.nrow+1)[::-1],
                xticklabels = np.arange(1, env.nrow+1))
    # Save as png
    env_name = env.spec.id
    plt.savefig(f"{env_name}-heatmap.png")
    # Other choices of cmap: YlGnBu
    # More: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    return None

@dataclass
class Args:
    mode: str = "policy_iteration_sync"
    gamma: float = 0.9
    theta: float = 1e-3

def main(args: Args):
    # Set numpy random seed
    np.random.seed(0)
    # Define num_trials, gamma and whatever variables you need below.
    envs = ['Deterministic-4x4-FrozenLake-v0', 'Deterministic-8x8-FrozenLake-v0']
    # Setup test function
    if args.mode == "policy_iteration_sync":
        eval_fn = policy_iteration_sync
    elif args.mode == "policy_iteration_async_ordered":
        eval_fn = policy_iteration_async_ordered
    elif args.mode == "policy_iteration_async_randperm":
        eval_fn = policy_iteration_async_randperm
    elif args.mode == "value_iteration_sync":
        eval_fn = value_iteration_sync
    elif args.mode == "value_iteration_async_ordered":
        eval_fn = value_iteration_async_ordered
    elif args.mode == "value_iteration_async_randperm":
        eval_fn = value_iteration_async_randperm
    elif args.mode == "value_iteration_async_custom":
        eval_fn = value_iteration_async_custom
    else:
        raise NotImplementedError(f"Unknown mode: {args.mode}")
    # Run the experiment
    for env_name in envs:
        env = env_wrapper(env_name)
        num_value_iter = -1
        num_policy_iter = -1
        if 'policy' in args.mode:
            if 'rand' in args.mode:
                # run 10 times and average
                policy = np.zeros([10, env.observation_space.n], dtype=np.int)
                value_func = np.zeros([10, env.observation_space.n])
                for i in range(10):
                    policy[i], value_func[i], num_policy_iter_current, num_value_iter_current = eval_fn(env, args.gamma, tol=args.theta)
                    num_value_iter += num_value_iter_current
                    num_policy_iter += num_policy_iter_current
                num_value_iter /= 10
                num_policy_iter /= 10
                policy = np.mean(policy, axis=0)
                value_func = np.mean(value_func, axis=0)
            else:
                policy, value_func, num_policy_iter, num_value_iter = eval_fn(env, args.gamma, tol=args.theta)
        elif 'value' in args.mode:
            if 'rand' in args.mode:
                # run 10 times and average
                value_func = np.zeros([10, env.observation_space.n])
                for i in range(10):
                    value_func[i], num_value_iter_current = eval_fn(env, args.gamma, tol=args.theta)
                    num_value_iter += num_value_iter_current
                num_value_iter /= 10
                value_func = np.mean(value_func, axis=0)
            else:
                value_func, num_value_iter = eval_fn(env, args.gamma, tol=args.theta)
            policy = value_function_to_policy(env, args.gamma, value_func)
        print('====================================')
        print(f"Policy for {env_name}")
        display_policy_letters(env, policy)
        print(f"Value function for {env_name}")
        print(value_func)
        value_func_heatmap(env, value_func)
        print(f"Number of iterations for {env_name}")
        print(f"Policy Iteration: {num_policy_iter}")
        print(f"Value Iteration: {num_value_iter}")
        print('====================================')

if __name__ == "__main__":
    main(tyro.cli(Args))