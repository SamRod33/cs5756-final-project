from agent import Agent
import numpy as np
from tqdm import tqdm

class MADDPG:
    """
    MADDPG (Multi Agent Deep Deterministic Policy Gradient) algorithm
    """
    def __init__(self, env, seed) -> None:
        self.env = env
        self.seed = seed
        self.agent_nets = {agent_id : Agent(env, agent_id, seed) for agent_id in env.agents}
        
    def run_episode(self):
        """
        Run an episode of the environment and return the episode

        Returns:
            episode (dict[str:list]): dictionary of list of tuples (state, action, reward)
        """
        obs, infos = self.env.reset(self.seed)
        self.env.reseed(self.seed)
        episode = {}
        while self.env.agents:
            actions = {agent: self.agent_nets[agent].select_action(obs[agent]) for agent in self.env.agents}
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            for agent in self.env.agents:
                sample = (obs[agent], actions[agent], rewards[agent])
                if agent in episode:
                    episode[agent].append(sample)
                else:
                    episode[agent] = [sample]
            obs = next_obs
        return episode
    
    def train(self, num_iterations, batch_size, gamma, lr):
        """Train each policy network and value network using the MADDPG algorithm

        Args:
            num_iterations (int): Number of iterations to train the policy and value networks
            batch_size (int): Number of episodes per batch
            gamma (float): Discount factor
            lr (float): Learning rate
        """
        avg_rewards = np.zeros(num_iterations) # average reward per episode, per agent
        for iter in tqdm(range(num_iterations)):
            episodes = [self.run_episode() for _ in range(batch_size)]
            for agent, agent_net in self.agent_nets.items():
                avg_agent_rewards = agent_net.train(episodes, num_iterations, batch_size, gamma, lr)
                avg_rewards += np.array(avg_agent_rewards)
        avg_rewards /= len(self.agent_nets)
        return avg_rewards