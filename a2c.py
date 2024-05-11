import torch
from torch import nn
from tqdm import tqdm
import numpy as np


class PolicyGradient:
    def __init__(self, env, policy_net, seed, reward_to_go: bool = False):
        """Policy gradient algorithm based on the REINFORCE algorithm.

        Args:
            env (gym.Env): Environment
            policy_net (PolicyNet): Policy network
            seed (int): Seed
            reward_to_go (bool): True if using reward_to_go, False if not
        """
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = policy_net.to(self.device)
        self.reward_to_go = reward_to_go
        self.seed = seed
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def select_action(self, state):
        """Select an action based on the policy network

        Args:
            state (np.ndarray): State of the environment

        Returns:
            action (int): Action to be taken
        """
        state_tensor = torch.tensor(state).to(self.device)
        action_probs = self.policy_net(state_tensor)
        action_categorical = torch.distributions.categorical.Categorical(action_probs)
        # TODO: make it deterministic
        action = action_categorical.sample().item()
        return action

    def compute_loss(self, episode, gamma):
        """Compute the loss function for the REINFORCE algorithm

        Args:
            episode (list): List of tuples (state, action, reward)
            gamma (float): Discount factor

        Returns:
            loss (torch.Tensor): The value of the loss function
        """
        episode_len = len(episode)
        rewards = torch.zeros(len(episode)).to(self.device)
        probs = torch.zeros(len(episode)).to(self.device)

        if not self.reward_to_go:
            for t, (state, action, reward) in enumerate(episode):
              rewards += (gamma ** t) * reward
        else:
            for t, (_, _, reward) in enumerate(reversed(episode)):
                rewards[:episode_len - t] *= gamma
                rewards[:episode_len - t] += reward

        for t, (state, action, reward) in enumerate(episode):
              probs[t] += self.policy_net(torch.tensor(state).to(self.device))[action]
        loss = -torch.sum(torch.log(probs) * rewards)

        return loss

    def update_policy(self, episodes, optimizer, gamma):
        """Update the policy network using the batch of episodes

        Args:
            episodes (list): List of episodes
            optimizer (torch.optim): Optimizer
            gamma (float): Discount factor
        """
        losses = []
        for episode in episodes:
            losses.append(self.compute_loss(episode, gamma))
        avg_loss = torch.mean(torch.stack(losses))

        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()


    # def run_episode(self):
    #     """
    #     Run an episode of the environment and return the episode

    #     Returns:
    #         episode (list): List of tuples (state, action, reward)
    #     """
    #     # TODO: will be moved to maddpg
    #     self.env.seed(self.seed)
    #     self.env.action_space.seed(self.seed)
    #     self.env.observation_space.seed(self.seed)
    #     state = self.env.reset()
    #     episode = []
    #     done = False
    #     while not done:
    #         action = self.select_action(state)
    #         next_state, reward, done, info = self.env.step(action)
    #         episode.append((state, action, reward))
    #         state = next_state
    #     return episode
    
    # we do not plan on using PolicyGradients directly, only as a superclass
    # def train(self, num_iterations, batch_size, gamma, lr):
    #     """Train the policy network using the REINFORCE algorithm

    #     Args:
    #         num_iterations (int): Number of iterations to train the policy network
    #         batch_size (int): Number of episodes per batch
    #         gamma (float): Discount factor
    #         lr (float): Learning rate
    #     """
    #     # TODO: take out episodes and ask as param and do not iterate here 
    #     avg_rewards = []
    #     self.policy_net.train()
    #     optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
    #     for iter in tqdm(range(num_iterations)):
    #         episodes = [self.run_episode() for _ in range(batch_size)]
    #         self.update_policy(episodes, optimizer, gamma)
    #         if iter % 10 == 0:
    #             avg_rewards.append(self.evaluate(10))
    #     return avg_rewards


    def evaluate(self, num_episodes = 100):
        """Evaluate the policy network by running multiple episodes.

        Args:
            num_episodes (int): Number of episodes to run

        Returns:
            average_reward (float): Average total reward per episode
        """
        self.policy_net.eval()
        total_rewards = []
        episodes = [self.run_episode() for _ in range(num_episodes)]
        for episode in episodes:
            rewards = 0
            for (_, _, reward) in episode:
                rewards += reward
            total_rewards.append(rewards)
        return np.mean(total_rewards)
    
    
class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """Policy network for the REINFORCE algorithm.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(in_features=state_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, state: torch.Tensor):
        """Forward pass of the policy network.

        Args:
            state (torch.Tensor): State of the environment.

        Returns:
            x (torch.Tensor): Probabilities of the actions.
        """
        z1 = self.relu(self.fc1(state))
        z2 = self.fc2(z1)
        z3 = self.softmax(z2)
        return z3


# TODO: make critic network global for all agents
class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        """Value network for the Actor-Critic algorithm.

        Args:
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(ValueNet, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_features=state_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor):
        """Forward pass of the value network.

        Args:
            state (torch.Tensor): State of the environment.

        Returns:
            x (torch.Tensor): Estimated value of the state.
        """
        y1 = self.fc1(state)
        z1 = self.relu(y1)
        z2 = self.fc2(z1)
        return z2



class ActorCriticPolicyGradient(PolicyGradient):
    def __init__(self, env, policy_net, value_net, seed, reward_to_go: bool = True):
        """A2C algorithm.

        Args:
            env (gym.Env): Environment
            policy_net (PolicyNet): Policy network
            value_net (ValueNet): Value network
            seed (int): Seed
            reward_to_go (bool): Not used
        """
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = policy_net.to(self.device)
        self.value_net = value_net.to(self.device)
        # TODO: fix seeding
        self.seed = seed
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def compute_loss(self, episode, gamma):
        """Compute the loss function for the A2C algorithm

        Args:
            episode (list): List of tuples (state, action, reward)

        Returns:
            policy_loss (torch.Tensor): Value of policy loss function
            value_loss (torch.Tensor): Value of value loss function
        """
        episode_len = len(episode)
        log_probs = torch.zeros(len(episode)).to(self.device)
        rewards = torch.zeros(len(episode)).to(self.device)
        critic_values = torch.zeros(len(episode), 1).to(self.device)
        advantage = torch.zeros(len(episode)).to(self.device)
        for t, (_, _, reward) in enumerate(reversed(episode)):
            rewards[:episode_len - t] *= gamma
            rewards[:episode_len - t] += reward
        for t, (state, action, reward) in enumerate(episode):
            state_tensor = torch.tensor(state).to(self.device)
            log_probs[t] += self.policy_net(state_tensor)[action]
            critic_values[t] += self.value_net(state_tensor)
        log_probs = torch.log(log_probs)
        critic_values = critic_values.squeeze()

        advantage += rewards - critic_values
        policy_loss = - torch.mean(log_probs * (advantage.detach()))
        value_loss = torch.mean(advantage * advantage)

        return policy_loss, value_loss

    def update_policy(self, episodes, optimizer, value_optimizer, gamma):
        """Update the policy network and value network using the batch of episodes

        Args:
            episodes (list): List of episodes
            optimizer (torch.optim): Optimizer for policy network
            value_optimizer (torch.optim): Optimizer for value network
            gamma (float): Discount factor
        """
        policy_losses = []
        value_losses = []
        for episode in episodes:
            policy_loss, value_loss = self.compute_loss(episode, gamma)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
        avg_policy_loss = torch.mean(torch.stack(policy_losses))
        avg_value_loss = torch.mean(torch.stack(value_losses))

        optimizer.zero_grad()
        avg_policy_loss.backward()
        optimizer.step()

        value_optimizer.zero_grad()
        avg_value_loss.backward()
        value_optimizer.step()


    def train(self, num_iterations, batch_size, gamma, lr):
        """Train the policy network and value network using the A2C algorithm

        Args:
            num_iterations (int): Number of iterations to train the policy and value networks
            batch_size (int): Number of episodes per batch
            gamma (float): Discount factor
            lr (float): Learning rate
        """
        # TODO: take out episodes and ask as param and do not iterate here
        avg_rewards = []
        self.policy_net.train()
        self.value_net.train()
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        for iter in tqdm(range(num_iterations)):
            episodes = [self.run_episode() for _ in range(batch_size)]
            self.update_policy(episodes, optimizer, value_optimizer, gamma)
            if iter % 10 == 0:
                avg_rewards.append(self.evaluate(10))
        return avg_rewards