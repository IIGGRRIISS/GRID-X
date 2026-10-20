"""
Strategy Optimiser – Wrapper for trained DQN agent.
Loads the agent and runs evaluation to produce pit strategy recommendations.
"""

import torch
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.models.race_env import RaceEnv
from scripts.models.dqn_agent import DQNAgent


class StrategyOptimiser:  # <-- Changed from StrategyOptimizer to StrategyOptimiser
    def __init__(self, agent_path=None):
        if agent_path is None:
            agent_path = project_root / 'models' / 'dqn_agent.pth'
        self.agent_path = agent_path
        self.agent = None
        self.state_size = 5  # lap_norm, tyre_age_norm, one-hot (3)
        self.action_size = 4

    def load(self):
        """Load the trained DQN agent."""
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.agent.load(self.agent_path)
        self.agent.epsilon = 0.0  # no exploration
        print("✅ Strategy Optimiser agent loaded.")

    def optimize(self, driver, circuit, weather, total_laps, start_compound='SOFT'):
        """
        Run the agent in evaluation mode to determine the optimal pit strategy.
        Returns a dict with total race time, pit stops (lap numbers), and compounds used.
        """
        if self.agent is None:
            self.load()

        # Create environment with given parameters
        env = RaceEnv(driver=driver, circuit=circuit, weather=weather,
                      total_laps=total_laps, start_compound=start_compound)

        state = env.reset()
        done = False
        pit_laps = []
        pit_compounds = []
        actions = []

        while not done:
            action = self.agent.act(state, eval_mode=True)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            if action != 0:  # pit
                pit_laps.append(env.current_lap)  # lap after pit
                pit_compounds.append(env.compounds[action - 1])
            state = next_state

        total_time = env.total_time

        return {
            'total_race_time': round(total_time, 2),
            'pit_stops': pit_laps,
            'pit_compounds': pit_compounds,
            'actions': actions  # optional, for debugging
        }