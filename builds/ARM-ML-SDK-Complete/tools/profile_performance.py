#!/usr/bin/env python3
"""
Performance profiler for ML operations with real-time metrics collection
"""

import subprocess
import time
import threading
import queue
import json
import sys
import os
from datetime import datetime

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class VulkanProfiler:
    """Performance profiler with real-time metrics collection"""

    def __init__(self):
        self.metrics = []
        self.metrics_queue = queue.Queue()
        self.monitoring = False
        self.metrics_history = []

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

        metric = {
            "name": name,
            "time_ms": (end - start) * 1000,
            "status": "success" if result.returncode == 0 else "failed",
            "timestamp": datetime.now().isoformat()
        }

        self.metrics.append(metric)
        return metric

    def start_realtime_profiling(self, scenario_path, name, duration=60, iterations=None):
        """Start real-time performance profiling with continuous metrics collection"""
        print(f"=== Real-time Performance Profiler ===")
        print(f"Profiling: {name}")
        print(f"Scenario: {scenario_path}")
        if iterations:
            print(f"Iterations: {iterations}")
        else:
            print(f"Duration: {duration} seconds")
        print("\nPress Ctrl+C to stop early\n")

        self.monitoring = True
        self.metrics_history = []

        # Start profiling thread
        profile_thread = threading.Thread(
            target=self._realtime_profile_loop,
            args=(scenario_path, name, duration, iterations)
        )
        profile_thread.start()

        # Display real-time metrics
        try:
            self._display_realtime_metrics()
        except KeyboardInterrupt:
            print("\nStopping profiler...")

        self.monitoring = False
        profile_thread.join()

        # Generate summary
        self._generate_realtime_summary(name)

    def _realtime_profile_loop(self, scenario_path, name, duration, iterations):
        """Profiling loop running in separate thread"""
        start_time = time.time()
        iteration = 0

        while self.monitoring:
            # Check termination conditions
            if iterations and iteration >= iterations:
                break
            if not iterations and (time.time() - start_time) >= duration:
                break

            iteration += 1

            # Run and measure
            metric = self._run_and_measure(scenario_path, name, iteration)
            self.metrics_queue.put(metric)
            self.metrics_history.append(metric)

            # Small delay between iterations
            time.sleep(0.05)

    def _run_and_measure(self, scenario_path, name, iteration):
        """Run scenario and measure performance"""
        start = time.perf_counter()

        # Run scenario
        result = subprocess.run([
            "../bin/scenario-runner",
            "--scenario", scenario_path,
            "--output", "/tmp/vulkan_profile_output",
            "--quiet"
        ], capture_output=True, env={"DYLD_LIBRARY_PATH": "/usr/local/lib"})

        end = time.perf_counter()
        execution_time_ms = (end - start) * 1000

        # Create metric
        metric = {
            "iteration": iteration,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "execution_time_ms": execution_time_ms,
            "success": result.returncode == 0,
            "fps": 1000 / execution_time_ms if execution_time_ms > 0 else 0
        }

        # Parse GPU metrics if available
        if result.returncode == 0 and result.stdout:
            try:
                output = result.stdout.decode()
                if "gpu_time" in output:
                    metric["gpu_time_ms"] = float(output.split("gpu_time:")[1].split()[0])
                if "memory_used" in output:
                    metric["memory_mb"] = float(output.split("memory_used:")[1].split()[0])
            except Exception:
                pass

        return metric

    def _display_realtime_metrics(self):
        """Display real-time metrics as they come in"""
        print("Iteration | Time (ms) | FPS   | Status")
        print("----------|-----------|-------|--------")

        while self.monitoring:
            try:
                metric = self.metrics_queue.get(timeout=1)
                status = "OK" if metric["success"] else "FAIL"
                print(f"{metric['iteration']:9d} | {metric['execution_time_ms']:9.2f} | {metric['fps']:5.1f} | {status}")
            except queue.Empty:
                continue

    def _generate_realtime_summary(self, name):
        """Generate summary from real-time profiling session"""
        if not self.metrics_history:
            print("\nNo metrics collected")
            return

        print(f"\n=== Performance Summary: {name} ===")

        # Calculate statistics
        times = [m["execution_time_ms"] for m in self.metrics_history if m["success"]]
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            print(f"Total iterations: {len(self.metrics_history)}")
            print(f"Successful runs: {len(times)}")
            print(f"Average execution time: {avg_time:.2f} ms")
            print(f"Min execution time: {min_time:.2f} ms")
            print(f"Max execution time: {max_time:.2f} ms")
            print(f"Average FPS: {1000/avg_time:.1f}")

            # Performance consistency
            if len(times) > 1:
                variance = sum((t - avg_time) ** 2 for t in times) / len(times)
                std_dev = variance ** 0.5
                print(f"Standard deviation: {std_dev:.2f} ms")
                print(f"Performance consistency: {100 - (std_dev/avg_time * 100):.1f}%")

            # Add to main metrics list
            self.metrics.append({
                "name": f"{name}_realtime",
                "time_ms": avg_time,
                "min_ms": min_time,
                "max_ms": max_time,
                "iterations": len(times),
                "status": "success"
            })
        else:
            print("No successful runs recorded")

        # Save detailed report
        report = {
            "operation": name,
            "summary": {
                "total_iterations": len(self.metrics_history),
                "successful_runs": len(times) if times else 0,
                "average_time_ms": avg_time if times else 0,
                "min_time_ms": min_time if times else 0,
                "max_time_ms": max_time if times else 0,
                "average_fps": 1000/avg_time if times else 0
            },
            "metrics": self.metrics_history
        }

        report_path = f"profile_{name}_realtime.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")

    def generate_report(self, output_path="performance_report"):
        """Generate performance report with optional visualization"""
        print("\n=== Performance Report ===")
        for metric in self.metrics:
            time_str = f"{metric['time_ms']:.2f} ms"
            if 'min_ms' in metric:
                time_str += f" (min: {metric['min_ms']:.2f}, max: {metric['max_ms']:.2f})"
            print(f"{metric['name']}: {time_str} ({metric['status']})")

        # Save JSON report
        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics": self.metrics
        }
        json_path = f"{output_path}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {json_path}")

        # Create visualization if matplotlib is available
        if HAS_MATPLOTLIB:
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
                png_path = f"{output_path}.png"
                plt.savefig(png_path)
                print(f"Visualization saved to: {png_path}")
        else:
            print("Note: Install matplotlib for visualization support")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Performance profiler for ML operations with real-time metrics collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scenario model.json --name conv2d
  %(prog)s --scenario model.json --name matmul --realtime --duration 30
  %(prog)s --scenario model.json --name pooling --realtime --iterations 100
  %(prog)s --batch  # Run batch profiling on default operations
        """
    )

    parser.add_argument("--scenario", "-s", help="Path to scenario file")
    parser.add_argument("--name", "-n", default="operation", help="Name for the operation being profiled")
    parser.add_argument("--realtime", "-r", action="store_true", help="Enable real-time metrics collection")
    parser.add_argument("--duration", "-d", type=int, default=60, help="Duration for real-time profiling (seconds)")
    parser.add_argument("--iterations", "-i", type=int, help="Number of iterations (overrides duration)")
    parser.add_argument("--batch", "-b", action="store_true", help="Run batch profiling on default operations")
    parser.add_argument("--output", "-o", default="performance_report", help="Output file prefix for reports")

    args = parser.parse_args()

    profiler = VulkanProfiler()

    if args.batch:
        # Batch mode: profile default operations
        operations = [
            ("conv2d", "../scenarios/conv2d_test.json"),
            ("matmul", "../scenarios/matmul_test.json"),
            ("pooling", "../scenarios/pooling_test.json")
        ]

        print("=== Batch Profiling Mode ===\n")
        for name, scenario in operations:
            if os.path.exists(scenario):
                print(f"Profiling {name}...")
                profiler.profile_operation(scenario, name)
            else:
                print(f"Skipping {name}: scenario file not found")

        profiler.generate_report(args.output)

    elif args.scenario:
        if not os.path.exists(args.scenario):
            print(f"Error: Scenario file not found: {args.scenario}")
            sys.exit(1)

        if args.realtime:
            # Real-time profiling mode
            profiler.start_realtime_profiling(
                args.scenario,
                args.name,
                duration=args.duration,
                iterations=args.iterations
            )
        else:
            # Single operation profiling
            print(f"Profiling {args.name}...")
            profiler.profile_operation(args.scenario, args.name)
            profiler.generate_report(args.output)

    else:
        parser.print_help()
        print("\nError: Either --scenario or --batch is required")
        sys.exit(1)


if __name__ == "__main__":
    main()
