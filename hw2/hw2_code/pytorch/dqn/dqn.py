#!/usr/bin/env python
import random
import numpy as np, gym, sys, copy, argparse
import os
import torch
import collections
import tqdm
import matplotlib.pyplot as plt

# set random seeds
torch.manual_seed(0)
np.random.seed(0)

class FullyConnectedModel(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(16, output_size)
        #no activation output layer

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr, logdir=None):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        # TODO Implement this method
        self.model = FullyConnectedModel(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        path = os.path.join(self.logdir, "model")
        torch.save(self.model.state_dict(), model_file)
        return path

    def load_model(self, model_file):
        # Helper function to load an existing model.
        return self.model.load_state_dict(torch.load(model_file))

    def load_model_weights(self,weight_file):
        # Optional Helper function to load model weights.
        pass


class Replay_Memory():
    def __init__(self, env, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        # Hint: you might find this useful:
        # 		collections.deque(maxlen=memory_size)
        # TODO Implement this method

        # Initialize episodes as a deque
        self.episodes = collections.deque(maxlen=memory_size)
        
        # Burn in episodes
        # Generate a bunch of (state, action, reward, next_state) tuples using random policy
        obs = env.reset()
        for i in range(burn_in):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            self.episodes.append((obs, action, reward, next_obs, done))
            obs = next_obs
            if done:
                obs = env.reset()

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        # TODO Implement this method
        
        # Sample a random batch of batch_size from self.episodes
        batch = random.sample(self.episodes, batch_size)
        return batch

    def append(self, transition):
        # Appends transition to the memory.
        # TODO Implement this method
        assert len(transition) == 5
        
        # Remove the oldest transition if memory is full
        if len(self.episodes) == self.episodes.maxlen:
            self.episodes.popleft()
        
        # Append the new transition
        self.episodes.append(transition)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, lr, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        # TODO Implement this method

        self.env = gym.make(environment_name)

        # Initialize QNetwork
        self.qn = QNetwork(self.env, lr)
        
        # Initialize memory
        self.memory = Replay_Memory(self.env)
        
        self.gamma = 0.99

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        # TODO Implement this method

        eps = 0.05
        if np.random.rand() < eps:
            action = np.random.choice(self.env.action_space.n)
        else:
            action = torch.argmax(q_values).item()
        return action

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        # TODO Implement this method
        
        action = torch.argmax(q_values).item()
        return action

    def train(self):
        # In this function, we will train our network.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        # TODO Implement this method

        self.qn.model.train()

        def update_network():
            # Sample a minibatch
            minibatch = self.memory.sample_batch()
            obs = torch.from_numpy(np.array([x[0] for x in minibatch]))
            action = torch.Tensor([x[1] for x in minibatch]).long()
            reward = torch.Tensor([x[2] for x in minibatch])
            next_obs = torch.from_numpy(np.array([x[3] for x in minibatch]))
            done = torch.Tensor([x[4] for x in minibatch])

            # Compute target and output Q values
            target_q = reward + self.gamma * torch.amax(target_qn(next_obs), dim=1) * (1 - done)
            output_q = self.qn.model(obs)[torch.arange(len(obs)), action]

            # Backprop
            loss = torch.nn.functional.mse_loss(output_q, target_q)
            self.qn.optimizer.zero_grad()
            loss.backward()
            self.qn.optimizer.step()

        num_episodes = 200
        timesteps_between_updates = 50
        timestep = 0
        target_qn = copy.deepcopy(self.qn.model)
        avg_returns = []
        for i in range(num_episodes):
            if i%10 == 0:
                avg_returns.append(self.test())

            obs = self.env.reset()
            done = False
            while not done:
                # Simulate one step and add it to memory
                action = self.epsilon_greedy_policy(self.qn.model(torch.from_numpy(obs).float()))
                next_obs, reward, done, _ = self.env.step(action)
                self.memory.append((obs, action, reward, next_obs, done))
                obs = next_obs

                # Do one minibatch of training
                update_network()

                # Replace target network every timesteps_between_updates
                timestep += 1
                if timestep % timesteps_between_updates == 0:
                    target_qn = copy.deepcopy(self.qn.model)
        return avg_returns

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 20 episodes, by calculating average cumulative rewards (returns) for the 20 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using replay memory.
        # TODO Implement this method
        self.qn.model.eval()

        total_ret = 0
        num_episodes = 20
        for i in range(num_episodes):
            obs = self.env.reset()
            done = False
            ret = 0
            while not done:
                action = self.greedy_policy(self.qn.model(torch.from_numpy(obs).float()))
                obs, reward, done, _ = self.env.step(action)
                ret += reward
            total_ret += ret
        avg_ret = total_ret / num_episodes
        return avg_ret

    # def burn_in_memory(self):
    #     # Initialize your replay memory with a burn_in number of episodes / transitions.
    #     # TODO Implement this method
    #     pass


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop.
def test_video(agent, env, epi):
    # Usage:
    # 	you can pass the arguments within agent.train() as:
    # 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4)
    return parser.parse_args()


def main(args):

    args = parse_arguments()
    environment_name = args.env

    returns_per_trial = []
    for i in range(5):
        print(f"Beginning trial {i}")
        agent = DQN_Agent(environment_name, args.lr, render=args.render)
        returns = agent.train() # (20,)
        returns_per_trial.append(returns)
    returns_per_trial = np.array(returns_per_trial) # (5, 20,)

    ks = np.arange(20)*10
    avs = np.mean(returns_per_trial, axis=0) # (20,)
    maxs = np.max(returns_per_trial, axis=0) # (20,)
    mins = np.min(returns_per_trial, axis=0) # (20,)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    if not os.path.exists('./plots'):
        os.mkdir('./plots')

    plt.title("DQN Learning Curve", fontsize = 24)
    plt.savefig("./plots/dqn_curve.png")

if __name__ == '__main__':
    main(sys.argv)
