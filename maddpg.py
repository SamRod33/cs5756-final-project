from agent import Agent
import numpy as np
from tqdm import tqdm

class MADDPG:
    """
    MADDPG (Multi Agent Deep Deterministic Policy Gradient) algorithm
    """
    def __init__(self, env, seed, num_iterations, batch_size, gamma, lr) -> None:
        self.env = env
        self.seed = seed
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.agent_nets = {agent_id : Agent(env, agent_id, seed, lr) for agent_id in env.agents}
        
    def run_episode(self):
        """
        Run an episode of the environment and return the episode

        Returns:
            episode (dict[str:list]): dictionary of list of tuples (state, action, reward)
        """
        obs, infos = self.env.reset(self.seed)
        self.env.reseed(self.seed)
        episode = {}
        max_steps = 100000
        i = 0
        while obs and i < max_steps:
            actions = {agent: self.agent_nets[agent].select_action(obs[agent].flatten()) for agent in obs}
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            for agent in obs:
                sample = (obs[agent].flatten(), actions[agent], rewards[agent])
                if agent in episode:
                    episode[agent].append(sample)
                else:
                    episode[agent] = [sample]
            i += 1
            obs = next_obs
        # print('FINISHED COLLECTING A BATCH')
        return episode
    
    def evaluate(self, num_episodes = 100):
        """Evaluate the policy network by running multiple episodes.

        Args:
            num_episodes (int): Number of episodes to run

        Returns:
            average_reward (float): Average total reward per episode
        """
        total_rewards = np.zeros(num_episodes)
        for agent_net in self.agent_nets.values():
            agent_net.policy_net.eval()
        episodes = [self.run_episode() for _ in range(num_episodes)]
        for t, episode_dict in enumerate(episodes):
            episode_rewards = []
            for agent, episode in episode_dict.items():
                rewards = 0
                for (_, _, reward) in episode:
                    rewards += reward
                episode_rewards.append(rewards)
            total_rewards[t] += np.mean(episode_rewards)
        return np.mean(total_rewards)
    
    def train(self):
        """Train each policy network and value network using the MADDPG algorithm

        Args:
            num_iterations (int): Number of iterations to train the policy and value networks
            batch_size (int): Number of episodes per batch
            gamma (float): Discount factor
            lr (float): Learning rate
        """
        avg_rewards = [] # average reward per 10th episode, per agent
        for iter in tqdm(range(self.num_iterations)):
            episodes = [self.run_episode() for _ in range(self.batch_size)]
            for agent, agent_net in self.agent_nets.items():
                agent_net.train(episodes, self.gamma)
            if iter % 10 == 0:
                avg_rewards.append(self.evaluate(10))
                # Open a file in write mode
                with open("output.txt", "w") as file:
                    # Iterate over the list
                    for item in avg_rewards:
                        # Write each item to the file
                        file.write(f"{item}\n")
        return avg_rewards
    
    def save_agents(self):
        for agent_net in self.agent_nets.values():
            agent_net.save()