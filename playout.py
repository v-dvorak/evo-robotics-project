import pickle
import time

import gymnasium as gym
import numpy as np

from fitness import Agent
import argparse


def load_genome(path: str):
    with open(path, "rb") as f:
        genome = pickle.load(f)

    return np.array(genome, dtype=np.float32)


def play(genome_path: str, max_steps: int, slow: bool = False):
    genome = load_genome(genome_path)

    agent = Agent(genome)

    env = gym.make(
        "CarRacing-v3",
        render_mode="human",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

    state, _ = env.reset()

    total_reward = 0.0

    for step in range(max_steps):
        action = agent.react(state)

        state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if step % 100 == 0:
            print(f"step={step} reward={total_reward:.2f}")

        if terminated or truncated:
            break

        # optional slow-down for easier viewing
        if slow:
            time.sleep(1 / 60)

    print(f"\nFinal reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Car Racing Playout")
    parser.add_argument("input")
    parser.add_argument("--slow", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)

    args = parser.parse_args()

    play(args.input, args.max_steps, slow=args.slow)
