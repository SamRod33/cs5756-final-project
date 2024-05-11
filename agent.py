from a2c import PolicyNet, ValueNet, ActorCriticPolicyGradient

HYPERPARAMS = {
    'value_net': {'hidden_dim': 128},
    'policy_net': {'hidden_dim': 128}
}

class Agent:
    def __init__(self, env, agent_id, seed) -> None:
        self.env = env
        self.agent_id = agent_id
        self.observation_space = env.observation_space(agent_id)
        self.action_space = env.action_space(agent_id)
        self.policy_net = PolicyNet(self.observation_space.shape[0], self.action_space.n, HYPERPARAMS['policy_net']['hidden_dim'])
        self.centralized_value_net = ValueNet(env.observation_space(agent_id).shape[0], HYPERPARAMS['value_net']['hidden_dim'])
        self.a2c = ActorCriticPolicyGradient(env, self.policy_net, self.centralized_value_net, seed)
        
    def select_action(self, state):
        """Select an action based on the policy network

        Args:
            state (np.ndarray): State of the environment

        Returns:
            action (int): Action to be taken
        """
        return self.a2c.select_action(state)
        
    def train(self, num_iterations, batch_size, gamma, lr):
        return self.a2c.train(num_iterations, batch_size, gamma, lr)
    
    def evaluate(self):
        return self.a2c.evaluate()