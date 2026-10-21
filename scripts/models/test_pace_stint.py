"""
Test full stint simulation using the trained LSTM.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.models.pace_forecaster import PaceForecaster

def test_stint():
    forecaster = PaceForecaster()
    forecaster.load_model()

    # Example: predict a 20-lap stint for VER at Monaco on Softs, dry weather
    weather = {'air_temp': 25, 'track_temp': 40, 'humidity': 60, 'rainfall': 0}
    laps = forecaster.predict_stint(
        driver_code='VER',
        circuit='Monaco',
        compound='SOFT',
        weather=weather,
        n_laps=20
    )

    print("\n🏁 Predicted stint for VER at Monaco (SOFT, dry):")
    for i, lap in enumerate(laps, 1):
        print(f"  Lap {i:2d}: {lap:.3f}s")

    # Plot the degradation curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(laps)+1), laps, marker='o', linestyle='-')
    plt.xlabel('Lap Number')
    plt.ylabel('Lap Time (s)')
    plt.title('Predicted Stint Degradation – VER, Monaco, SOFT')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_stint()