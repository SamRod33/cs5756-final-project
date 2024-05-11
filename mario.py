from pettingzoo.atari import mario_bros_v3
import time


SEED = 695

def mario_env(seed=SEED, render=False):
    return MarioEnv(seed, render)


class MarioEnv:
    def __init__(self, seed=SEED, render=False) -> None:
        rom_path = '.'
        self.env = mario_bros_v3.parallel_env(auto_rom_install_path=rom_path, render_mode=('human' if render else None))
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
        for agent_id in self.agents:
            self.observation_space(agent_id).seed(seed)
            self.action_space(agent_id).seed(seed)
        
    
        
        
    
if __name__ == '__main__':
    env = mario_env(render=True)
    observations, infos = env.reset()

    while env.agents:
        time.sleep(0.01)
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()