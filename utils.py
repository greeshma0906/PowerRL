import datetime

def log_energy_usage(energy, log_file="logs/energy_log.txt"):
    """Logs energy consumption to a file."""
    with open(log_file, "a") as f:
        f.write(f"{datetime.datetime.now()} - Energy Used: {energy:.3f} kWh\n")

def read_logs(log_file="logs/energy_log.txt"):
    """Reads the log file."""
    with open(log_file, "r") as f:
        return f.readlines()
