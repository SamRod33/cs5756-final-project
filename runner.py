import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from maddpg import MADDPG
import mario

seed = mario.SEED
num_iterations = 11
batch_size = 1
gamma = 0.99
lr = 0.01

# Setting the seed to ensure reproducability
def reseed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def plot_avg_reward(avg_rewards):
    # TODO: EDIT
    # iterations = (np.arange(20) + 1) * 10
    iterations = np.arange(num_iterations // 10 + 1) * 10
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, avg_rewards, marker='o', linestyle='-', color='b')
    plt.title('MADDPG Advantage Actor-Critic Average Rewards During Training')
    plt.xlabel('Training Iteration')
    # plt.xticks(range(0, 210, 10))
    plt.ylabel('Average Reward')
    plt.savefig('plot.png')
    plt.show()

if __name__ == '__main__':
    reseed(seed)

    # Feel free to use the space below to run experiments and create plots used in your writeup.
    env = mario.mario_env()
    env.reseed()
    algo = MADDPG(env, seed, num_iterations, batch_size, gamma, lr)
    avg_rewards = algo.train()
    plot_avg_reward(avg_rewards)
    algo.save_agents()
    

    