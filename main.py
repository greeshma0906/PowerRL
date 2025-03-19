import subprocess
import sys
from energy_monitor import EnergyMonitor

def run_rl_script(script_path):
    """Runs the RL script while monitoring its energy consumption."""
    monitor = EnergyMonitor()
    
    # Start tracking energy before execution
    monitor.start()

    # Execute the RL script
    process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Stop tracking after execution
    monitor.stop()

    # Print script output
    print(stdout.decode())
    if stderr:
        print("Errors:", stderr.decode())

    # Display energy consumption
    energy_used = monitor.get_energy_usage()
    print(f"Total Energy Consumed: {energy_used:.3f} kWh")

if __name__ == "_main_":
    rl_script = "rl_scripts/sample_rl.py"  # Replace with actual script path
    run_rl_script(rl_script)