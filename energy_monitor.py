import time
import psutil  # For CPU energy tracking

class EnergyMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.cpu_start = None
        self.cpu_end = None

    def get_cpu_power(self):
        """Estimate CPU power usage in watts"""
        return psutil.cpu_percent() * 0.2  # Approximate power per CPU utilization percentage

    def start(self):
        self.start_time = time.time()
        self.cpu_start = self.get_cpu_power()

    def stop(self):
        self.end_time = time.time()
        self.cpu_end = self.get_cpu_power()

    def get_energy_usage(self):
        duration = (self.end_time - self.start_time) / 3600  # Convert to hours
        avg_cpu_power = (self.cpu_start + self.cpu_end) / 2
        energy_consumed = avg_cpu_power * duration  # kWh
        return energy_consumed

if __name__ == "__main__":
    monitor = EnergyMonitor()
    monitor.start()
    time.sleep(5)  # Simulate workload
    monitor.stop()
    print(f"Energy Used: {monitor.get_energy_usage():.3f} kWh")
