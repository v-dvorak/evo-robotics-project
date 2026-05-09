import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib

if len(sys.argv) < 2:
    print('should be called like $ python plot_visual_field.py vision.cross')
    exit()

vision = importlib.import_module(sys.argv[1])
PIXELS = vision.PIXELS

pixels_x, pixels_y = zip(*PIXELS)


env = gym.make("CarRacing-v3", render_mode="rgb_array")
state, _ = env.reset()

for _ in range(100):
    action = np.zeros(3)
    state, reward, terminated, truncated, _ = env.step(action)


dir_path = Path('visual_field')
dir_path.mkdir(parents=True, exist_ok=True)

plt.imsave(dir_path / f'{sys.argv[1]}_state.png', state)

state[pixels_x, pixels_y, :] = 255
plt.imsave(dir_path / f'{sys.argv[1]}_points.png', state)

state[:, :, 0] = 0
state[:, :, 2] = 0
state[pixels_x, pixels_y, :] = 255
plt.imsave(dir_path / f'{sys.argv[1]}_green-points.png', state)
