import numpy as np

## TensorFlow Only ##
# import tensorflow as tf
# import keras
# from model_tensorflow import make_model


## Pytorch Only ##
import torch
from model_pytorch import make_model, ExpertModel


def action_to_one_hot(env, action):
    action_vec = np.zeros(env.action_space.n)
    action_vec[action] = 1
    return action_vec


def generate_episode(env, policy):
    """Collects one rollout from the policy in an environment. The environment
    should implement the OpenAI Gym interface. A rollout ends when done=True. The
    number of states and actions should be the same, so you should not include
    the final state when done=True.

    Args:
    env: an OpenAI Gym environment.
    policy: The output of a deep neural network
    Returns:
    states: a list of states visited by the agent.
    actions: a list of actions taken by the agent. For tensorflow, it will be
        helpful to use a one-hot encoding to represent discrete actions. The actions
        that you return should be one-hot vectors (use action_to_one_hot()).
        For Pytorch, the Cross-Entropy Loss function will integers for action
        labels.
    rewards: the reward received by the agent at each step.
    """
    done = False
    # Chaoyi Change
    state = env.reset()
    # state, _ = env.reset()

    states = []
    actions = []
    rewards = []
    while not done:
        # append the state, action, and reward to the lists
        states.append(state)
        action_tensor = policy(torch.from_numpy(state).float())
        action_tensor = torch.softmax(action_tensor, dim=0)
        action = np.argmax(action_tensor.detach().numpy())
        action_one_hot = action_to_one_hot(env, action)
        actions.append(action_one_hot)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    return states, actions, rewards


class Imitation:
    def __init__(self, env, num_episodes, expert_file):
        self.env = env

        # TensorFlow Only #
        # self.expert = tf.keras.models.load_model(expert_file)

        # Pytorch Only #
        self.expert = ExpertModel()
        self.expert.load_state_dict(torch.load(expert_file))
        self.expert.eval()

        self.num_episodes = num_episodes
        self.model = make_model()

    def generate_behavior_cloning_data(self):
        self._train_states = []
        self._train_actions = []
        for _ in range(self.num_episodes):
            states, actions, _ = generate_episode(self.env, self.expert)
            self._train_states.extend(states)
            self._train_actions.extend(actions)
        self._train_states = np.array(self._train_states)
        self._train_actions = np.array(self._train_actions)

    def generate_dagger_data(self):
        # WRITE CODE HERE
        # You should collect states and actions from the student policy
        # (self.model), and then relabel the actions using the expert policy.
        # This method does not return anything.
        self._train_states = []
        self._train_actions = []
        for _ in range(self.num_episodes):
            states, actions, _ = generate_episode(self.env, self.model)
            for s in states:
                action_tensor = self.expert(torch.from_numpy(s).float())
                action_tensor = torch.softmax(action_tensor, dim=0)
                action = np.argmax(action_tensor.detach().numpy())
                action_one_hot = action_to_one_hot(self.env, action)
                self._train_states.append(s)
                self._train_actions.append(action_one_hot)
        self._train_states = np.array(self._train_states)
        self._train_actions = np.array(self._train_actions)
        # END

    def train(self, num_epochs=1, batch_size=64):
        """
        Train the model on data generated by the expert policy.
        Use Cross-Entropy Loss and a batch size of 64 when
        performing updates.
        Args:
            num_epochs: number of epochs to train on the data generated by the expert.
        Return:
            loss: (float) final loss of the trained policy.
            acc: (float) final accuracy of the trained policy
        """
        # WRITE CODE HERE
        # train related
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        # data related
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self._train_states).float(),
            torch.from_numpy(self._train_actions).float(),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # train
        total_loss = 0
        cnt = 0
        for ep in range(num_epochs):
            for states, actions in dataloader:
                # forward prop
                logits = self.model(states)
                loss = loss_fn(logits, actions)
                total_loss += loss.item()
                cnt += 1
                # backward prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss = total_loss / cnt

        # evaluate on the trained dataset
        num_correct = 0
        num_data = 0
        for states, actions in dataloader:
            action_pred = torch.softmax(self.model(states), dim=1)
            action_pred = torch.argmax(action_pred, dim=1)
            action_expert = torch.argmax(actions, dim=1)
            num_correct += (action_pred == action_expert).sum().item()
            num_data += actions.shape[0]
        acc = num_correct / num_data

        # END
        return loss, acc

    def evaluate(self, policy, n_episodes=50):
        rewards = []
        for i in range(n_episodes):
            _, _, r = generate_episode(self.env, policy)
            rewards.append(sum(r))
        r_mean = np.mean(rewards)
        return r_mean