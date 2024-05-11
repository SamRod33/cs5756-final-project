import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from a2c import PolicyGradient, ActorCriticPolicyGradient, ValueNet, PolicyNet

seed = 695

# Setting the seed to ensure reproducability
def reseed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

reseed(seed)

# Feel free to use the space below to run experiments and create plots used in your writeup.
env = gym.make("CartPole-v1")
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

# Feel free to use the space below to run experiments and create plots used in your writeup.
reseed(seed)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
env.reset()

policy_net_a2c = PolicyNet(env.observation_space.shape[0], env.action_space.n, 128)
value_net = ValueNet(env.observation_space.shape[0], 128)

a2c = ActorCriticPolicyGradient(env, policy_net_a2c, value_net, seed)
avg_rewards_a2c = a2c.train(num_iterations=200, batch_size=10, gamma=0.99, lr=0.01)

# visualize(algorithm=a2c, video_name="a2c")
print(f'\nACTOR-CRITIC Expected Reward: {a2c.evaluate()}')

iterations = (np.arange(20) + 1) * 10
plt.figure(figsize=(10, 6))
plt.plot(iterations, avg_rewards_a2c, marker='o', linestyle='-', color='b')
plt.title('Advantage Actor-Critic (Part 3) Average Rewards During Training')
plt.xlabel('Training Iteration')
plt.xticks(range(0, 210, 10))
plt.ylabel('Average Reward')
plt.savefig('plot-p3.png')

plt.show()