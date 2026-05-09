from vision.cross import PIXELS

pixels_x, pixels_y = zip(*PIXELS)

from pathlib import Path
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

env = gym.make("CarRacing-v3", render_mode="rgb_array")
state, _ = env.reset()

for _ in range(100):
    action = np.zeros(3)
    state, reward, terminated, truncated, _ = env.step(action)


dir_path = Path('visual_field')
dir_path.mkdir(parents=True, exist_ok=True)

plt.imsave(dir_path / 'state.png', state)

state[pixels_x, pixels_y, :] = 255
plt.imsave(dir_path / 'points.png', state)

state[:, :, 0] = 0
state[:, :, 2] = 0
state[pixels_x, pixels_y, :] = 255
plt.imsave(dir_path / 'green-points.png', state)
