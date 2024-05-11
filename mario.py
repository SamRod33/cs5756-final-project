from pettingzoo.atari import mario_bros_v3
import time

rom_path = '.'

env = mario_bros_v3.parallel_env(auto_rom_install_path=rom_path, render_mode='human')
observations, infos = env.reset()

while env.agents:
    time.sleep(0.01)
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()