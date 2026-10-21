"""
Train DQN agent on the race environment with variable race length.
"""

import numpy as np
import torch
from race_env import RaceEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def train(episodes=1000):  # increased episodes to cover more variability
    # Environment parameters (constant weather, driver, circuit)
    weather = {'air_temp': 25, 'track_temp': 40, 'humidity': 60, 'rainfall': 0}
    driver = 'VER'
    circuit = 'Monaco'
    start_compound = 'SOFT'

    # We'll create a new environment each episode with random total_laps
    # but we need to know state_size (depends on number of compounds, which is fixed)
    temp_env = RaceEnv(driver=driver, circuit=circuit, weather=weather,
                       total_laps=20, start_compound=start_compound)
    state_size = 2 + len(temp_env.compounds)
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    rewards_history = []
    time_history = []

    for ep in range(episodes):
        # Random race length between 20 and 80 laps
        total_laps = random.randint(20, 80)

        env = RaceEnv(driver=driver, circuit=circuit, weather=weather,
                      total_laps=total_laps, start_compound=start_compound)

        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        agent.update_epsilon()
        if (ep + 1) % agent.target_update == 0:
            agent.update_target_network()

        race_time = -total_reward
        rewards_history.append(race_time)
        time_history.append(env.total_time)

        if (ep + 1) % 50 == 0:
            avg_time = np.mean(time_history[-50:])
            print(f"Episode {ep+1}/{episodes}, Avg Race Time: {avg_time:.2f}s, Epsilon: {agent.epsilon:.3f}")

    # Save trained agent
    save_path = Path(__file__).parent.parent.parent / 'models' / 'dqn_agent.pth'
    agent.save(save_path)
    print(f"✅ Agent saved to {save_path}")

    # Plot learning curve
    plt.figure(figsize=(10,5))
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Race Time (s)')
    plt.title('DQN Training Progress (Variable Race Length)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train(episodes=1000)  # more episodes because task is harder