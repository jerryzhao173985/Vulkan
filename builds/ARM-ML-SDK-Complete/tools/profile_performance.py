#!/usr/bin/env python3
"""Performance profiler for ML operations"""

import subprocess
import time
import json
import matplotlib.pyplot as plt

class VulkanProfiler:
    def __init__(self):
        self.metrics = []
    
    def profile_operation(self, scenario_path, name):
        """Profile a single operation"""
        start = time.perf_counter()
        
        # Run scenario
        result = subprocess.run([
            "../bin/scenario-runner",
            "--scenario", scenario_path,
            "--output", ".",
            "--profiling-dump-path", f"profile_{name}.json"
        ], capture_output=True, env={"DYLD_LIBRARY_PATH": "/usr/local/lib"})
        
        end = time.perf_counter()
        
        self.metrics.append({
            "name": name,
            "time_ms": (end - start) * 1000,
            "status": "success" if result.returncode == 0 else "failed"
        })
    
    def generate_report(self):
        """Generate performance report"""
        print("\n=== Performance Report ===")
        for metric in self.metrics:
            print(f"{metric['name']}: {metric['time_ms']:.2f} ms ({metric['status']})")
        
        # Create visualization
        names = [m['name'] for m in self.metrics if m['status'] == 'success']
        times = [m['time_ms'] for m in self.metrics if m['status'] == 'success']
        
        if names:
            plt.figure(figsize=(10, 6))
            plt.bar(names, times)
            plt.xlabel('Operation')
            plt.ylabel('Time (ms)')
            plt.title('ML Operation Performance on Apple Silicon')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('performance_report.png')
            print("\nVisualization saved to performance_report.png")

if __name__ == "__main__":
    profiler = VulkanProfiler()
    
    # Profile different operations
    operations = [
        ("conv2d", "../scenarios/conv2d_test.json"),
        ("matmul", "../scenarios/matmul_test.json"),
        ("pooling", "../scenarios/pooling_test.json")
    ]
    
    for name, scenario in operations:
        if os.path.exists(scenario):
            profiler.profile_operation(scenario, name)
    
    profiler.generate_report()
