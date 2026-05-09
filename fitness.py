import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = None


def make_env():
    global env
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95,
                   domain_randomize=False, continuous=True)
    env.reset(seed=42)


# environment
MAX_STEPS = 1000

# we consider only green channel
IM = 96
MID = IM//2
CHANNEL = 1
PIXELS = [
    (MID, MID),
    (MID-10, MID),
    (MID+10, MID),
    (MID, MID-5),
    (MID, MID+5),
]
pixels_x, pixels_y = zip(*PIXELS)

# neural network
SIZE_INPUT = len(PIXELS)
SIZE_HIDDEN = 24

param_shapes = [
    # hidden layer (weights & biases)
    (SIZE_HIDDEN, SIZE_INPUT),
    (SIZE_HIDDEN,),
    # output layer
    (3, SIZE_HIDDEN),
    (3,),
]
param_sizes = [int(np.prod(s)) for s in param_shapes]
param_splits = np.cumsum(param_sizes[:-1]).tolist()
genome_len = sum(param_sizes)


class Agent:
    def __init__(self, genome):
        param_parts = np.split(genome, param_splits)
        self._W1, self._b1, self._W2, self._b2 = (
            p.reshape(s) for p, s in zip(param_parts, param_shapes)
        )
        # action buffer to save time
        self.action = np.zeros(3)

    def react(self, state: np.ndarray) -> np.ndarray:
        x = state[pixels_x, pixels_y, CHANNEL] / 255
        h = np.tanh(self._W1 @ x + self._b1)
        out = np.tanh(self._W2 @ h + self._b2)
        self.action[0] = out[0]  # steering (-1, 1)
        self.action[1] = (out[1] + 1.0) * 0.5  # gas (0, 1)
        self.action[2] = (out[2] + 1.0) * 0.5  # braking (0, 1)
        return self.action


def evaluate(genome):
    agent = Agent(genome)
    total_reward = 0.0
    state, _ = env.reset()

    for _ in range(MAX_STEPS):
        # state[pixels_x, pixels_y, :] = 255
        # plt.imshow(state)
        # plt.show()

        action = agent.react(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward,
