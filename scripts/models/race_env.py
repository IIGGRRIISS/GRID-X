"""
Race Environment for Reinforcement Learning
Simulates a race with compound‑specific linear degradation.
"""

import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.models.int_en_pred_2 import GridXIntegratedPredictor


class RaceEnv:
    def __init__(self, driver, circuit, weather, total_laps, start_compound='SOFT'):
        self.driver = driver
        self.circuit = circuit
        self.weather = weather
        self.total_laps = total_laps
        self.start_compound = start_compound

        # Load predictor only once (to get first lap time)
        if not hasattr(RaceEnv, 'predictor'):
            print("🔄 Loading GRID‑X integrated predictor for baseline lap time...")
            RaceEnv.predictor = GridXIntegratedPredictor()
            RaceEnv.predictor.load_or_train_models()
            print("✅ Predictor loaded.")

        self.pit_loss = 20.0
        self.compounds = ['SOFT', 'MEDIUM', 'HARD']
        self.compound_to_idx = {c: i for i, c in enumerate(self.compounds)}
        
        # Compound‑specific degradation rates (seconds per lap)
        self.degradation_rates = {
            'SOFT': 0.12,
            'MEDIUM': 0.08,
            'HARD': 0.05
        }

        # Get baseline first lap time (for the first stint)
        self.first_lap_time = self._get_first_lap_time()
        print(f"ℹ️ Baseline first lap time: {self.first_lap_time:.3f}s")

        self.reset()

    def _get_first_lap_time(self):
        """Use the predictor to get a realistic first lap time."""
        driver_num_str = str(RaceEnv.predictor.driver_number_map.get(self.driver, 0))
        feat = {
            'DriverNumber': driver_num_str,
            'LapNumber': 1,
            'Stint': 1,
            'Compound': self.start_compound,
            'Team': 'UNKNOWN',
            'event_name': f"{self.circuit} Grand Prix",
            'circuit': self.circuit,
            'year': 2024,
            'round': 1,
            'AirTemp': self.weather['air_temp'],
            'Humidity': self.weather['humidity'],
            'Pressure': 1013.0,
            'Rainfall': self.weather['rainfall'],
            'TrackTemp': self.weather['track_temp'],
            'WindSpeed': 5.0,
            'WindDirection': 180,
            'stint_lap_number': 1,
            'tyre_age_laps': 1,
            'session_progress': 1 / self.total_laps,
            'Position': 1,
            'position_change': 0
        }
        return RaceEnv.predictor.predict_lap_with_features(feat)

    def reset(self):
        self.current_lap = 0
        self.tyre_age = 1
        self.current_compound = self.start_compound
        self.stint = 1
        self.done = False
        self.total_time = 0.0
        return self._get_state()

    def _get_state(self):
        lap_norm = self.current_lap / self.total_laps
        tyre_age_norm = self.tyre_age / 30.0   # max tyre age for normalisation
        one_hot = np.zeros(len(self.compounds))
        one_hot[self.compound_to_idx[self.current_compound]] = 1
        state = np.concatenate([[lap_norm, tyre_age_norm], one_hot])
        return state.astype(np.float32)

    def _lap_time(self):
        """Linear degradation based on tyre age and current compound."""
        base = self.first_lap_time
        rate = self.degradation_rates[self.current_compound]
        return base + (self.tyre_age - 1) * rate

    def step(self, action):
        if self.done:
            raise ValueError("Episode already finished. Call reset() first.")

        reward = 0
        if action == 0:
            # Stay out
            lap_time = self._lap_time()
            reward -= lap_time
            self.total_time += lap_time
            self.current_lap += 1
            self.tyre_age += 1
        else:
            # Pit to new compound
            new_compound = self.compounds[action - 1]
            self.total_time += self.pit_loss
            reward -= self.pit_loss
            self.current_compound = new_compound
            self.tyre_age = 1
            self.stint += 1
            # Out‑lap (degradation applies with new compound)
            lap_time = self._lap_time()
            reward -= lap_time
            self.total_time += lap_time
            self.current_lap += 1

        if self.current_lap >= self.total_laps:
            self.done = True

        next_state = self._get_state()
        return next_state, reward, self.done, {}


if __name__ == "__main__":
    weather = {'air_temp': 25, 'track_temp': 40, 'humidity': 60, 'rainfall': 0}
    env = RaceEnv(driver='VER', circuit='Monaco', weather=weather,
                  total_laps=20, start_compound='SOFT')
    state = env.reset()
    print("Initial state:", state)

    while not env.done:
        action = np.random.randint(0, 4)
        next_state, reward, done, _ = env.step(action)
        print(f"Lap {env.current_lap}: action={action}, reward={reward:.3f}, total_time={env.total_time:.3f}")
        state = next_state
