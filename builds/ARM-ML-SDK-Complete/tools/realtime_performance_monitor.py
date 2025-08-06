#!/usr/bin/env python3
"""
Real-time performance monitoring for Vulkan ML workloads
"""

import subprocess
import time
import threading
import queue
import json
import sys
from datetime import datetime

class VulkanPerformanceMonitor:
    def __init__(self):
        self.metrics_queue = queue.Queue()
        self.monitoring = False
        self.metrics_history = []
        
    def start_monitoring(self, scenario_path, duration=60):
        """Start real-time performance monitoring"""
        print(f"=== Real-time Performance Monitor ===")
        print(f"Monitoring: {scenario_path}")
        print(f"Duration: {duration} seconds")
        print("\nPress Ctrl+C to stop early\n")
        
        self.monitoring = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(scenario_path, duration)
        )
        monitor_thread.start()
        
        # Display real-time metrics
        try:
            self._display_metrics()
        except KeyboardInterrupt:
            print("\nStopping monitor...")
        
        self.monitoring = False
        monitor_thread.join()
        
        # Generate report
        self._generate_report()
    
    def _monitor_loop(self, scenario_path, duration):
        """Monitor loop running in separate thread"""
        start_time = time.time()
        iteration = 0
        
        while self.monitoring and (time.time() - start_time) < duration:
            iteration += 1
            
            # Run scenario and measure performance
            metric = self._run_and_measure(scenario_path, iteration)
            self.metrics_queue.put(metric)
            self.metrics_history.append(metric)
            
            # Small delay between iterations
            time.sleep(0.1)
    
    def _run_and_measure(self, scenario_path, iteration):
        """Run scenario and measure performance"""
        start = time.perf_counter()
        
        # Run scenario
        result = subprocess.run([
            "../bin/scenario-runner",
            "--scenario", scenario_path,
            "--output", "/tmp/vulkan_output",
            "--quiet"
        ], capture_output=True, env={"DYLD_LIBRARY_PATH": "/usr/local/lib"})
        
        end = time.perf_counter()
        
        # Create metric
        metric = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "execution_time_ms": (end - start) * 1000,
            "success": result.returncode == 0,
            "fps": 1000 / ((end - start) * 1000) if (end - start) > 0 else 0
        }
        
        # Parse GPU metrics if available
        if result.returncode == 0 and result.stdout:
            try:
                output = result.stdout.decode()
                # Extract metrics from output
                if "gpu_time" in output:
                    metric["gpu_time_ms"] = float(output.split("gpu_time:")[1].split()[0])
                if "memory_used" in output:
                    metric["memory_mb"] = float(output.split("memory_used:")[1].split()[0])
            except:
                pass
        
        return metric
    
    def _display_metrics(self):
        """Display real-time metrics"""
        print("Iteration | Time (ms) | FPS   | Status")
        print("----------|-----------|-------|--------")
        
        while self.monitoring:
            try:
                metric = self.metrics_queue.get(timeout=1)
                status = "OK" if metric["success"] else "FAIL"
                print(f"{metric['iteration']:9d} | {metric['execution_time_ms']:9.2f} | {metric['fps']:5.1f} | {status}")
            except queue.Empty:
                continue
    
    def _generate_report(self):
        """Generate performance report"""
        if not self.metrics_history:
            print("\nNo metrics collected")
            return
        
        print("\n=== Performance Summary ===")
        
        # Calculate statistics
        times = [m["execution_time_ms"] for m in self.metrics_history if m["success"]]
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"Average execution time: {avg_time:.2f} ms")
            print(f"Min execution time: {min_time:.2f} ms")
            print(f"Max execution time: {max_time:.2f} ms")
            print(f"Average FPS: {1000/avg_time:.1f}")
            
            # Performance consistency
            variance = sum((t - avg_time) ** 2 for t in times) / len(times)
            std_dev = variance ** 0.5
            print(f"Standard deviation: {std_dev:.2f} ms")
            print(f"Performance consistency: {100 - (std_dev/avg_time * 100):.1f}%")
        
        # Save detailed report
        report = {
            "summary": {
                "total_iterations": len(self.metrics_history),
                "successful_runs": len(times),
                "average_time_ms": avg_time if times else 0,
                "min_time_ms": min_time if times else 0,
                "max_time_ms": max_time if times else 0
            },
            "metrics": self.metrics_history
        }
        
        with open("performance_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: performance_report.json")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-time performance monitor")
    parser.add_argument("scenario", help="Path to scenario file")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    
    args = parser.parse_args()
    
    monitor = VulkanPerformanceMonitor()
    monitor.start_monitoring(args.scenario, args.duration)

if __name__ == "__main__":
    main()
