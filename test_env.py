import gymnasium as gym
import safety_gymnasium
import numpy as np

make_kwargs = {'render_mode': 'rgb_array', 'width': 64, 'height': 64}
env = safety_gymnasium.make('SafetyPointGoal1-v0', **make_kwargs, camera_name='vision')
env.reset()

total_reward = 0
for i in range(100):
    _, r, cost, term, trunc, info = env.step(env.action_space.sample())
    total_reward += r
print(f"Average random step reward: {total_reward/100:.4f}")
