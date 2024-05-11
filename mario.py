from pettingzoo.atari import mario_bros_v3
import time

def mario_env():
    return MarioEnv()

SEED = 695

class MarioEnv:
    def __init__(self, seed=SEED) -> None:
        rom_path = '.'
        self.env = mario_bros_v3.parallel_env(auto_rom_install_path=rom_path, render_mode='human')
        self.reset(seed)
        self.agents = self.env.agents
        self.num_agents = self.env.num_agents
        self.possible_agents = self.env.possible_agents
        self.max_num_agents = self.env.max_num_agents
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        
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
        for agent_id in self.agents:
            self.observation_space(agent_id).seed(seed)
            self.action_space(agent_id).seed(seed)
        
    
        
        
    
if __name__ == '__main__':
    env = mario_env()
    observations, infos = env.reset()

    while env.agents:
        time.sleep(0.01)
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()