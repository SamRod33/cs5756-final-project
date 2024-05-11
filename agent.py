from a2c import PolicyNet, ValueNet, ActorCriticPolicyGradient
import torch

HYPERPARAMS = {
    'value_net': {'hidden_dim': 128},
    'policy_net': {'hidden_dim': 128}
}

class Agent:
    def __init__(self, env, agent_id, seed, lr) -> None:
        self.env = env
        self.agent_id = agent_id
        self.observation_space = env.observation_space(agent_id)
        self.action_space = env.action_space(agent_id)
        self.policy_net = PolicyNet(self.env.obs_dim(agent_id), self.action_space.n, HYPERPARAMS['policy_net']['hidden_dim'])
        self.centralized_value_net = ValueNet(self.env.global_obs_dim(), HYPERPARAMS['value_net']['hidden_dim'])
        self.a2c = ActorCriticPolicyGradient(env, self.policy_net, self.centralized_value_net, seed, lr)
        
    def select_action(self, state):
        """Select an action based on the policy network

        Args:
            state (np.ndarray): State of the environment

        Returns:
            action (int): Action to be taken
        """
        return self.a2c.select_action(state)
        
    def train(self, episodes, gamma):
        return self.a2c.train(episodes, gamma, self.agent_id)
    
    def evaluate(self, num_episodes=100):
        return self.a2c.evaluate(num_episodes)
    
    def save(self):
        torch.save(self.policy_net.state_dict(), f'{self.agent_id}-policy.pt')
    
    def load(self, path):
        self.policy_net = PolicyNet(self.env.obs_dim(self.agent_id), self.action_space.n, HYPERPARAMS['policy_net']['hidden_dim'])
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()