from collections import OrderedDict 
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
from queue import Queue
# relative import make_model from model_pytorch.py
from model_pytorch import make_model
import torch
from tqdm import trange
import tyro
from dataclasses import dataclass
# Import make_model here from the approptiate model_*.py file
# This model should be the same as problem 2

### 2.1 Build Goal-Conditioned Task
class FourRooms:
    def __init__(self, l=5, T=30):
        '''
        FourRooms Environment for pedagogic purposes
        Each room is a l*l square gridworld, 
        connected by four narrow corridors,
        the center is at (l+1, l+1).
        There are two kinds of walls:
        - borders: x = 0 and 2*l+2 and y = 0 and 2*l+2 
        - central walls
        T: maximum horizion of one episode
            should be larger than O(4*l)
        '''
        assert l % 2 == 1 and l >= 5
        self.l = l
        self.total_l = 2 * l + 3
        self.T = T

        # create a map: zeros (walls) and ones (valid grids)
        self.map = np.ones((self.total_l, self.total_l), dtype=bool)
        # build walls
        self.map[0, :] = self.map[-1, :] = self.map[:, 0] = self.map[:, -1] = False
        self.map[l+1, [1,2,-3,-2]] = self.map[[1,2,-3,-2], l+1] = False
        self.map[l+1, l+1] = False

        # define action mapping (go right/up/left/down, counter-clockwise)
        # e.g [1, 0] means + 1 in x coordinate, no change in y coordinate hence
        # hence resulting in moving right
        self.act_set = np.array([
            [1, 0], [0, 1], [-1, 0], [0, -1] 
        ], dtype=int)
        self.action_space = spaces.Discrete(4)

        # you may use self.act_map in search algorithm 
        self.act_map = {}
        self.act_map[(1, 0)] = 0
        self.act_map[(0, 1)] = 1
        self.act_map[(-1, 0)] = 2
        self.act_map[(0, -1)] = 3

    def render_map(self):
        plt.imshow(self.map)
        plt.xlabel('y')
        plt.ylabel('x')
        plt.savefig('p2_map.png', 
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()
    
    def sample_sg(self):
        # sample s
        while True:
            s = [np.random.randint(self.total_l), 
                np.random.randint(self.total_l)]
            if self.map[s[0], s[1]]:
                break

        # sample g
        while True:
            g = [np.random.randint(self.total_l), 
                np.random.randint(self.total_l)]
            if self.map[g[0], g[1]] and \
                (s[0] != g[0] or s[1] != g[1]):
                break
        return s, g

    def reset(self, s=None, g=None):
        '''
        Args:
            s: starting position, np.array((2,))
            g: goal, np.array((2,))
        Return:
            obs: np.cat(s, g)
        '''
        if s is None or g is None:
            s, g = self.sample_sg()
        else:
            assert 0 < s[0] < self.total_l - 1 and 0 < s[1] < self.total_l - 1
            assert 0 < g[0] < self.total_l - 1 and 0 < g[1] < self.total_l - 1
            assert (s[0] != g[0] or s[1] != g[1])
            assert self.map[s[0], s[1]] and self.map[g[0], g[1]]
        
        self.s = s
        self.g = g
        self.t = 1

        return self._obs()
    
    def step(self, a):
        '''
        Args:
            a: action, index into act_set
        Return obs, reward, done, info:
            done: whether the state has reached the goal
            info: succ if the state has reached the goal, fail otherwise 
        '''
        assert self.action_space.contains(a)

        # WRITE CODE HERE

        # transition to next state
        next_state = [self.s[0] + self.act_set[a][0], self.s[1] + self.act_set[a][1]]
        # check if next state is valid
        stay_inside_x = (0 < next_state[0] < self.total_l - 1)
        stay_inside_y = (0 < next_state[1] < self.total_l - 1)
        stay_inside_map = self.map[next_state[0], next_state[1]]
        if stay_inside_x and stay_inside_y and stay_inside_map:
            self.s = next_state
            reward = 0.0
        else:
            reward = -1.0
            
        # reward, info and done
        reach_goal = (self.s[0] == self.g[0] and self.s[1] == self.g[1])
        reach_time_limit = (self.t == self.T)
        done = reach_goal or reach_time_limit
        info = 'succ' if reach_goal else 'fail'
        reward = 1.0 if reach_goal else reward
        self.t += 1
        
        return self._obs(), 0.0, done, info

    def _obs(self):
        return np.concatenate([self.s, self.g])


def plot_traj(env, ax, traj, goal=None):
    traj_map = env.map.copy().astype(np.float32)
    traj_map[traj[:, 0], traj[:, 1]] = 2 # visited states
    traj_map[traj[0, 0], traj[0, 1]] = 1.5 # starting state
    traj_map[traj[-1, 0], traj[-1, 1]] = 2.5 # ending state
    if goal is not None:
        traj_map[goal[0], goal[1]] = 3 # goal
    ax.imshow(traj_map)
    ax.set_xlabel('y')
    ax.set_label('x')

### A uniformly random policy's trajectory
def test_step(env):
    s = np.array([1, 1])
    g = np.array([2*l+1, 2*l+1])
    s = env.reset(s, g)
    done = False
    traj = [s]
    while not done:
        s, _, done, _ = env.step(env.action_space.sample())
        traj.append(s)
    traj = np.array(traj)

    ax = plt.subplot()
    plot_traj(env, ax, traj, g)
    plt.savefig('p2_random_traj.png', 
            bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

def shortest_path_expert(env):
    """ 
    Implement a shortest path algorithm and collect N trajectories for N goal reaching tasks
    """
    N = 1000
    expert_trajs = []
    expert_actions = []

    # WRITE CODE HERE
    # use BFS to find the shortest path

    def BFS_search(env, start, goal):
        # queue for frontier
        frontier = Queue()
        frontier.put(start)

        # map current node to its parent
        came_from = OrderedDict()
        came_from[tuple(start)] = None # NOTE start is a list

        # do BFS
        while not frontier.empty():
            current = frontier.get()
            if current[0] == goal[0] and current[1] == goal[1]:
                break
            for a in range(4):
                next_state = [current[0] + env.act_set[a][0], current[1] + env.act_set[a][1]]
                stay_inside_x = (0 < next_state[0] < env.total_l - 1)
                stay_inside_y = (0 < next_state[1] < env.total_l - 1)
                stay_inside_map = env.map[next_state[0], next_state[1]]
                if stay_inside_x and stay_inside_y and stay_inside_map and (tuple(next_state) not in came_from):
                    frontier.put(next_state)
                    came_from[tuple(next_state)] = current, a

        # get path and related actions
        path = []
        current = goal
        while current != start:
            path.append((current, came_from[tuple(current)][1]))
            current = came_from[tuple(current)][0]
        
        # reverse path
        path.reverse()

        return path
    
    # generate expert trajectories
    for i in range(N):
        # sample a start and goal
        s, g = env.sample_sg()
        # find shortest path
        path = BFS_search(env, s, g)
        # convert path to trajectory
        traj = [np.concatenate([s, g])]
        actions = []
        for state in path:
            traj.append(np.concatenate([state[0], g]))
            actions.append(state[1])
        traj = np.array(traj)
        actions = np.array(actions)
        # append to list
        expert_trajs.append(traj)
        expert_actions.append(actions)


    # END
    # You should obtain expert_trajs, expert_actions from search algorithm

    fig, axes = plt.subplots(5,5, figsize=(10,10))
    axes = axes.reshape(-1)
    for idx, ax in enumerate(axes):
        plot_traj(env, ax, expert_trajs[idx])
    
    # Plot a subset of expert state trajectories
    plt.savefig('p2_expert_trajs.png', 
            bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    return expert_trajs, expert_actions


class GCBC:

    def __init__(self, env, expert_trajs, expert_actions):
        self.env = env
        self.expert_trajs = expert_trajs
        self.expert_actions = expert_actions
        self.transition_num = sum(map(len, expert_actions))
        self.model = make_model(input_dim=4, output_dim=4)
        # state_dim + goal_dim = 4
        # action_choices = 4

        # Initialize Optimizer here
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def generate_behavior_cloning_data(self):
        # training state should be a concatenation of state and goal
        self._train_states = []
        self._train_actions = []
        
        # WRITE CODE HERE
        for traj, actions in zip(self.expert_trajs, self.expert_actions):
            # append expert experience into training data
            for state, action in zip(traj, actions):
                self._train_states.append(state)
                self._train_actions.append(action)
        
        # END

        self._train_states = np.array(self._train_states).astype(np.float32) # size: (*, 4)
        self._train_actions = np.array(self._train_actions) # size: (*, )
        
    def generate_relabel_data(self):
        # apply expert data goal relabelling trick
        self._train_states = []
        self._train_actions = []

        # WRITE CODE HERE
        for traj, actions in zip(self.expert_trajs, self.expert_actions):
            # append expert experience into training data
            for i, (state, action) in enumerate(zip(traj, actions)):
                # set goal state to a random selected future state in the same trajectory
                goal_idx = np.random.randint(i+1, len(traj))
                # for goal_idx in np.arange(i+1, len(traj)):
                goal = traj[goal_idx][:2]
                # concatenate state and goal to get training state
                training_state = np.concatenate([state[:2], goal])
                self._train_states.append(training_state)
                # append action to training action
                self._train_actions.append(action)
        # END

        self._train_states = np.array(self._train_states).astype(np.float32) # size: (*, 4)
        self._train_actions = np.array(self._train_actions) # size: (*, 4)

    def train(self, num_epochs=20, batch_size=256):
        """ 
        Trains the model on training data generated by the expert policy.
        Args:
            num_epochs: number of epochs to train on the data generated by the expert.
            batch_size
        Return:
            loss: (float) final loss of the trained policy.
            acc: (float) final accuracy of the trained policy
        """
        # WRITE CODE HERE
        total_loss = 0.0
        total_acc = 0.0
        cnt = 0
        for ep in range(num_epochs):
            # shuffle data
            idx = np.arange(len(self._train_states))
            np.random.shuffle(idx)
            # train in batches
            for i in range(0, len(self._train_states), batch_size):
                # get batch
                batch_idx = idx[i:i+batch_size]
                batch_states = self._train_states[batch_idx]
                batch_actions = self._train_actions[batch_idx]
                # convert to tensor
                batch_states = torch.from_numpy(batch_states)
                batch_actions = torch.from_numpy(batch_actions)
                # forward pass
                logits = self.model(batch_states)
                # compute accuracy NOTE: use sum instead of mean to avoid bool to float conversion
                acc = torch.sum(torch.argmax(torch.softmax(logits, dim=1), dim=1) == batch_actions).item() / len(batch_actions)
                total_acc += acc
                # compute loss
                loss = torch.nn.functional.cross_entropy(logits, batch_actions)
                total_loss += loss.item()
                # counter
                cnt += 1
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # END
        return total_loss / cnt, total_acc / cnt


def evaluate_gc(env, policy, n_episodes=50):
    succs = 0
    for _ in range(n_episodes):
        info = generate_gc_episode(env, policy)
        # WRITE CODE HERE
        if info == 'succ':
            succs += 1
        # END
    succs /= n_episodes
    return succs

def generate_gc_episode(env, policy):
    """Collects one rollout from the policy in an environment. The environment
    should implement the OpenAI Gym interface. A rollout ends when done=True. The
    number of states and actions should be the same, so you should not include
    the final state when done=True.
    Args:
        env: an OpenAI Gym environment.
        policy: a trained model
    Returns:
    """
    done = False
    state = env.reset()
    while not done:
        # WRITE CODE HERE

        # run policy
        state = torch.FloatTensor(state)
        logits = policy.model(state)
        action = torch.argmax(torch.softmax(logits, dim=0)).item()

        # step env
        state, _, done, info = env.step(action)

        # END

    return info

def generate_random_trajs(env):
    N = 1000
    random_trajs = []
    random_actions = []

    # WRITE CODE HERE
    for i in range(N):
        # reset environment
        s, g = env.sample_sg()
        obs = env.reset(s, g)
        traj, acts = [obs], []
        # generate random trajectory
        while True:
            # sample a random action
            a = np.random.randint(4)
            # transition to next state
            obs, rew, done, info = env.step(a)
            # append to trajectory
            traj.append(obs)
            acts.append(a)
            # check if done
            if done:
                break
        # append to list
        random_trajs.append(np.array(traj))
        random_actions.append(np.array(acts))


    # END
    # You should obtain random_trajs, random_actions from random policy
    # train GCBC based on the previous code
    # WRITE CODE HERE

    return random_trajs, random_actions

@dataclass 
class Args:
    sample_mode: str = "expert" # "expert" or "random"
    mode: str = "vanilla" # "vanilla" or "relabel"
    exp: str = "gcbc" # "gcbc" or "render" or "expert"


def run_GCBC(args: Args):
    sample_mode = args.sample_mode
    mode = args.mode
    num_seeds = 5
    loss_vecs = []
    acc_vecs = []
    succ_vecs = []

    for i in range(num_seeds):
        print('*' * 50)
        print('seed: %d' % i)
        loss_vec = []
        acc_vec = []
        succ_vec = []

        # generate new set of trajectories
        # obtain either expert or random trajectories
        print('generating trajectories...')
        if sample_mode == "expert":
            expert_trajs, expert_actions = shortest_path_expert(env)
        elif sample_mode == "random":
            expert_trajs, expert_actions = generate_random_trajs(env)
        else:
            raise NotImplementedError
        gcbc = GCBC(env, expert_trajs, expert_actions)

        if mode == "vanilla":
            gcbc.generate_behavior_cloning_data()
        elif mode == "relabel":
            gcbc.generate_relabel_data()
        else:
            raise NotImplementedError

        print('training...')
        iter_num = 150 if args.sample_mode == "expert" else 50
        for e in trange(iter_num):
            loss, acc = gcbc.train(num_epochs=20)
            succ = evaluate_gc(env, gcbc)
            loss_vec.append(loss)
            acc_vec.append(acc)
            succ_vec.append(succ)
            print('ep', e, ' loss', round(loss,3), ' acc', round(acc,3), ' succ', succ)
        loss_vecs.append(loss_vec)
        acc_vecs.append(acc_vec)
        succ_vecs.append(succ_vec)

    loss_vec = np.mean(np.array(loss_vecs), axis = 0).tolist()
    acc_vec = np.mean(np.array(acc_vecs), axis = 0).tolist()
    succ_vec = np.mean(np.array(succ_vecs), axis = 0).tolist()

    ### Plot the results
    print('plotting...')
    from scipy.ndimage import uniform_filter
    # you may use uniform_filter(succ_vec, 5) to smooth succ_vec
    succ_vec_smooth = uniform_filter(succ_vec, 5)
    plt.figure(figsize=(12, 3))
    # WRITE CODE HERE
    # plot success rate, training loss and training accuracy
    plt.subplot(1, 3, 1)
    plt.plot(succ_vec_smooth)
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.title('Success Rate')
    plt.subplot(1, 3, 2)
    plt.plot(loss_vec)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.subplot(1, 3, 3)
    plt.plot(acc_vec)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Accuracy')
    # END
    plt.savefig(f'p2_gcbc_{mode}_{sample_mode}.png', dpi=300)
    plt.show()


# build env
l, T = 5, 30
env = FourRooms(l, T)
### Visualize the map
args = tyro.cli(Args)
if args.exp == "render":
    env.render_map()
elif args.exp == "expert":
    shortest_path_expert(env)
elif args.exp == "gcbc":
    run_GCBC(args)
else:
    raise NotImplementedError