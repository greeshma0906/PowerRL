# import time
# import numpy as np

# def train_rl_agent(epochs=10):
#     """Simulate an RL training process."""
#     for epoch in range(epochs):
#         state = np.random.rand(4)
#         action = np.argmax(state)  # Fake action selection
#         reward = np.random.rand()  # Fake reward
#         print(f"Epoch {epoch}: State {state}, Action {action}, Reward {reward}")
#         time.sleep(1)  # Simulate computation

# if __name__ == "__main__":
#     train_rl_agent()
import time
import random

class SimpleEnergyMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()

    def get_energy_usage(self):
        """Estimate energy usage based on execution time."""
        duration = (self.end_time - self.start_time) / 3600  # Convert seconds to hours
        estimated_power = 50  # Assume an average CPU power of 50W
        energy_consumed = estimated_power * duration  # Energy in kWh
        return energy_consumed

# Testing the monitor with a dummy computation
if __name__ == "__main__":
    monitor = SimpleEnergyMonitor()
    
    print("Starting energy monitoring...")
    monitor.start()

    # Simulated RL training (dummy computation)
    for i in range(7):
        data = [random.random() for _ in range(1000000)]  # Heavy computation
        max_value = max(data)  # Find max value (simulating RL computation)
        print(f"Iteration {i+1}: Max Value = {max_value:.5f}")
        time.sleep(1)  # Simulate processing time

    monitor.stop()
    print("Energy monitoring stopped.")

    # Display estimated energy consumption
    energy_used = monitor.get_energy_usage()
    print(f"Estimated Energy Consumed: {energy_used:.6f} kWh")
