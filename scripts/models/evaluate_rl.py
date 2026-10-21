"""
Evaluate trained DQN agent on the race environment.
"""

import sys
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.models.race_env import RaceEnv
from scripts.models.dqn_agent import DQNAgent

def evaluate(agent_path, episodes=5):
    # Environment parameters (same as training)
    weather = {'air_temp': 25, 'track_temp': 40, 'humidity': 60, 'rainfall': 0}
    env = RaceEnv(driver='VER', circuit='Monaco', weather=weather,
                  total_laps=20, start_compound='SOFT')
    state_size = 2 + len(env.compounds)   # lap_norm, tyre_age_norm, one-hot
    action_size = 4

    agent = DQNAgent(state_size, action_size)
    agent.load(agent_path)
    agent.epsilon = 0.0  # no exploration

    compound_names = ['Stay Out', 'SOFT', 'MEDIUM', 'HARD']

    for ep in range(episodes):
        state = env.reset()
        done = False
        actions = []
        pit_laps = []
        pit_compounds = []
        stint = 1
        current_compound = env.current_compound

        print(f"\n🏁 Episode {ep+1}")

        while not done:
            action = agent.act(state, eval_mode=True)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)

            # Record pit stops
            if action != 0:   # pitting
                pit_laps.append(env.current_lap)   # lap after pit (the lap we just completed)
                pit_compounds.append(compound_names[action])

            state = next_state

        total_time = env.total_time
        print(f"   Total race time: {total_time:.2f}s")
        if pit_laps:
            print(f"   Pit stops at laps: {pit_laps}")
            print(f"   New compounds: {pit_compounds}")
        else:
            print("   No pit stops (one‑stint race)")

if __name__ == "__main__":
    agent_path = project_root / 'models' / 'dqn_agent.pth'
    evaluate(agent_path, episodes=5)