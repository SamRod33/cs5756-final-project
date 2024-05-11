from pettingzoo.atari import mario_bros_v3
import time
import numpy as np
from supersuit import color_reduction_v0, frame_stack_v1, dtype_v0, resize_v1
from agent import Agent

SEED = 695

def mario_env(seed=SEED, render=False):
    return MarioEnv(seed, render)


class MarioEnv:
    def __init__(self, seed=SEED, render=False) -> None:
        rom_path = '.'
        self.env = mario_bros_v3.parallel_env(auto_rom_install_path=rom_path, render_mode=('human' if render else None))
        self.env = dtype_v0(resize_v1(color_reduction_v0(self.env, mode='R'), 105, 80, linear_interp=False), np.float32)
        self.reset(seed)
        self.agents = self.env.agents
        self.num_agents = self.env.num_agents
        self.possible_agents = self.env.possible_agents
        self.max_num_agents = self.env.max_num_agents
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.reseed(seed)
    
    def step(self, actions):
        return self.env.step(actions)
    
    def reset(self, seed=None, options=None):
        return self.env.reset(seed, options)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def state(self):
        return self.env.state()
    
    def observation_space(self, agent_id):
        return self.env.observation_space(agent_id)
    
    def action_space(self, agent_id):
        return self.env.action_space(agent_id)
    
    def reseed(self, seed=SEED):
        self.reset(seed)
        for agent_id in self.agents:
            self.observation_space(agent_id).seed(seed)
            self.action_space(agent_id).seed(seed)
            
    def obs_dim(self, agent_id):
        return np.prod(self.observation_space(agent_id).shape)
    
    def global_obs_dim(self):
        dim = 0
        for agent in self.env.agents:
            dim += self.obs_dim(agent)
        return dim
    
def untrained_sim():
    env = mario_env(render=True)
    observations, infos = env.reset()
    
    while env.agents:
        time.sleep(0.01)
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()
    
def trained_sim():
    env = mario_env(render=True)
    observations, infos = env.reset()
    agent_nets = {agent_id : Agent(env, agent_id, SEED, lr=0.01) for agent_id in env.agents}
    for agent_id in env.agents:
        agent_nets[agent_id].load(f'{agent_id}-policy.pt')

    while env.agents:
        time.sleep(0.01)
        actions = {agent: agent_nets[agent].select_action(observations[agent].flatten()) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()
        
    
if __name__ == '__main__':
    trained_sim()