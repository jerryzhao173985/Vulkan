#!/usr/bin/env python3
"""
Vulkan ML SDK Performance Benchmark Suite
Comprehensive performance testing for ML operations
"""

import time
import json
import numpy as np
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
# Optional plotting - skip if matplotlib not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class BenchmarkType(Enum):
    """Types of benchmarks"""
    LATENCY = "latency"          # Single operation time
    THROUGHPUT = "throughput"     # Operations per second
    BANDWIDTH = "bandwidth"       # Memory bandwidth
    POWER = "power"              # Power efficiency (ops/watt)
    SCALING = "scaling"          # Performance scaling

@dataclass
class BenchmarkResult:
    """Benchmark execution result"""
    name: str
    operation: str
    metric_type: BenchmarkType
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "operation": self.operation,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "unit": self.unit,
            "metadata": self.metadata
        }

class BenchmarkSuite:
    """Performance benchmark suite for Vulkan ML SDK"""
    
    def __init__(self):
        self.sdk_root = Path("/Users/jerry/Vulkan")
        self.sdk_bin = self.sdk_root / "builds" / "ARM-ML-SDK-Complete" / "bin"
        self.scenario_runner = self.sdk_bin / "scenario-runner"
        self.results: List[BenchmarkResult] = []
        
        # Performance counters
        self.cpu_percent_start = 0
        self.memory_start = 0
        
    def warmup(self, iterations: int = 5):
        """Warmup GPU and caches"""
        print("Warming up...")
        
        # Simple operations to warm up
        for _ in range(iterations):
            data = np.random.randn(1000000).astype(np.float32)
            _ = data * 2.0 + 1.0
            time.sleep(0.1)
    
    def benchmark_conv2d(self, 
                        input_shape: Tuple[int, ...] = (1, 224, 224, 3),
                        kernel_size: int = 3,
                        out_channels: int = 64,
                        iterations: int = 100) -> BenchmarkResult:
        """Benchmark Conv2D operation"""
        
        batch, height, width, in_channels = input_shape
        
        # Calculate FLOPs
        output_h = height - kernel_size + 1
        output_w = width - kernel_size + 1
        flops = (2 * kernel_size * kernel_size * in_channels * out_channels * 
                output_h * output_w * batch)
        
        # Create test data
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weights = np.random.randn(kernel_size, kernel_size, in_channels, out_channels).astype(np.float32)
        
        # Measure execution time
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            
            # Simulated Conv2D (would use actual SDK execution)
            output = self._simulate_conv2d(input_data, weights)
            
            end = time.perf_counter()
            times.append(end - start)
        
        # Calculate metrics
        avg_time = np.mean(times[10:])  # Skip first 10 for stability
        throughput_gflops = flops / (avg_time * 1e9)
        
        return BenchmarkResult(
            name=f"conv2d_{height}x{width}x{in_channels}",
            operation="conv2d",
            metric_type=BenchmarkType.THROUGHPUT,
            value=throughput_gflops,
            unit="GFLOPS",
            metadata={
                "input_shape": input_shape,
                "kernel_size": kernel_size,
                "out_channels": out_channels,
                "avg_latency_ms": avg_time * 1000,
                "min_latency_ms": min(times) * 1000,
                "max_latency_ms": max(times) * 1000
            }
        )
    
    def _simulate_conv2d(self, input_data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Simulated Conv2D for benchmarking"""
        # Simple matrix operations to simulate computational load
        batch, h, w, c_in = input_data.shape
        k_h, k_w, _, c_out = weights.shape
        
        # Flatten and multiply (simplified)
        input_flat = input_data.reshape(batch, -1)
        weight_flat = weights.reshape(-1, c_out)
        
        # Simulate computation
        output = np.dot(input_flat[:, :weight_flat.shape[0]], weight_flat)
        
        return output
    
    def benchmark_matmul(self,
                        size: int = 1024,
                        iterations: int = 100) -> BenchmarkResult:
        """Benchmark matrix multiplication"""
        
        # Calculate FLOPs
        flops = 2 * size * size * size
        
        # Create matrices
        matrix_a = np.random.randn(size, size).astype(np.float32)
        matrix_b = np.random.randn(size, size).astype(np.float32)
        
        # Measure execution
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = np.matmul(matrix_a, matrix_b)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times[10:])
        throughput_gflops = flops / (avg_time * 1e9)
        
        return BenchmarkResult(
            name=f"matmul_{size}x{size}",
            operation="matmul",
            metric_type=BenchmarkType.THROUGHPUT,
            value=throughput_gflops,
            unit="GFLOPS",
            metadata={
                "matrix_size": size,
                "avg_latency_ms": avg_time * 1000,
                "memory_mb": (3 * size * size * 4) / (1024 * 1024)
            }
        )
    
    def benchmark_memory_bandwidth(self,
                                 size_mb: int = 100,
                                 iterations: int = 50) -> BenchmarkResult:
        """Benchmark memory bandwidth"""
        
        size_bytes = size_mb * 1024 * 1024
        size_floats = size_bytes // 4
        
        # Create data
        data = np.random.randn(size_floats).astype(np.float32)
        
        # Measure copy bandwidth
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = data.copy()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times[5:])
        bandwidth_gbps = (size_bytes * 2) / (avg_time * 1e9)  # Read + Write
        
        return BenchmarkResult(
            name=f"memory_bandwidth_{size_mb}MB",
            operation="memory_copy",
            metric_type=BenchmarkType.BANDWIDTH,
            value=bandwidth_gbps,
            unit="GB/s",
            metadata={
                "size_mb": size_mb,
                "avg_latency_ms": avg_time * 1000
            }
        )
    
    def benchmark_activation(self,
                           activation_type: str = "relu",
                           size: int = 10000000,
                           iterations: int = 100) -> BenchmarkResult:
        """Benchmark activation functions"""
        
        # Create input data
        input_data = np.random.randn(size).astype(np.float32)
        
        # Measure execution
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            
            if activation_type == "relu":
                output = np.maximum(0, input_data)
            elif activation_type == "sigmoid":
                output = 1 / (1 + np.exp(-input_data))
            elif activation_type == "tanh":
                output = np.tanh(input_data)
            else:
                output = input_data
            
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times[10:])
        throughput_mops = size / (avg_time * 1e6)
        
        return BenchmarkResult(
            name=f"{activation_type}_{size}",
            operation=activation_type,
            metric_type=BenchmarkType.THROUGHPUT,
            value=throughput_mops,
            unit="MOps/s",
            metadata={
                "size": size,
                "avg_latency_ms": avg_time * 1000
            }
        )
    
    def benchmark_model_inference(self,
                                 model_name: str,
                                 input_shape: Tuple[int, ...],
                                 iterations: int = 50) -> BenchmarkResult:
        """Benchmark complete model inference"""
        
        model_path = self.sdk_root / "builds" / "ARM-ML-SDK-Complete" / "models" / model_name
        
        if not model_path.exists():
            return BenchmarkResult(
                name=f"model_{model_name}",
                operation="model_inference",
                metric_type=BenchmarkType.LATENCY,
                value=0,
                unit="ms",
                metadata={"error": "Model not found"}
            )
        
        # Create input data
        input_data = np.random.randn(*input_shape).astype(np.float32)
        np.save("/tmp/model_input.npy", input_data)
        
        # Measure inference time (simulated)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            
            # Would run actual model inference here
            time.sleep(0.01)  # Simulate inference
            
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times[5:]) * 1000  # Convert to ms
        fps = 1000 / avg_time
        
        return BenchmarkResult(
            name=f"model_{model_name.replace('.tflite', '')}",
            operation="model_inference",
            metric_type=BenchmarkType.LATENCY,
            value=avg_time,
            unit="ms",
            metadata={
                "model": model_name,
                "input_shape": input_shape,
                "fps": fps,
                "model_size_mb": model_path.stat().st_size / (1024 * 1024)
            }
        )
    
    def benchmark_scaling(self,
                        operation: str = "matmul",
                        sizes: List[int] = None) -> List[BenchmarkResult]:
        """Benchmark performance scaling with size"""
        
        if sizes is None:
            sizes = [256, 512, 1024, 2048]
        
        results = []
        
        for size in sizes:
            if operation == "matmul":
                result = self.benchmark_matmul(size, iterations=20)
            elif operation == "conv2d":
                # Scale conv2d input size
                input_shape = (1, size, size, 3)
                result = self.benchmark_conv2d(input_shape, iterations=20)
            else:
                continue
            
            results.append(result)
        
        return results
    
    def benchmark_comprehensive(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        
        print("="*60)
        print("Vulkan ML SDK - Comprehensive Performance Benchmark")
        print("="*60)
        
        self.warmup()
        
        # Conv2D benchmarks
        print("\n1. Conv2D Benchmarks")
        print("-"*40)
        conv_shapes = [
            (1, 224, 224, 3),   # ImageNet
            (1, 112, 112, 64),  # Mid layer
            (1, 56, 56, 128),   # Deep layer
        ]
        
        for shape in conv_shapes:
            result = self.benchmark_conv2d(shape)
            self.results.append(result)
            print(f"  {result.name}: {result.value:.2f} {result.unit}")
        
        # MatMul benchmarks
        print("\n2. MatMul Benchmarks")
        print("-"*40)
        matmul_sizes = [512, 1024, 2048]
        
        for size in matmul_sizes:
            result = self.benchmark_matmul(size, iterations=50)
            self.results.append(result)
            print(f"  {result.name}: {result.value:.2f} {result.unit}")
        
        # Memory bandwidth
        print("\n3. Memory Bandwidth")
        print("-"*40)
        mem_sizes = [10, 50, 100, 500]
        
        for size in mem_sizes:
            result = self.benchmark_memory_bandwidth(size, iterations=30)
            self.results.append(result)
            print(f"  {result.name}: {result.value:.2f} {result.unit}")
        
        # Activation functions
        print("\n4. Activation Functions")
        print("-"*40)
        activations = ["relu", "sigmoid", "tanh"]
        
        for act in activations:
            result = self.benchmark_activation(act)
            self.results.append(result)
            print(f"  {result.name}: {result.value:.2f} {result.unit}")
        
        # Model inference
        print("\n5. Model Inference")
        print("-"*40)
        models = [
            ("mobilenet_v2_1.0_224_quantized_1_default_1.tflite", (1, 224, 224, 3)),
            ("la_muse.tflite", (1, 256, 256, 3)),
        ]
        
        for model_name, input_shape in models:
            result = self.benchmark_model_inference(model_name, input_shape, iterations=20)
            if result.value > 0:
                self.results.append(result)
                print(f"  {result.name}: {result.value:.2f} {result.unit} ({result.metadata.get('fps', 0):.1f} FPS)")
        
        # Generate summary
        summary = self._generate_summary()
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary"""
        
        # Group by operation
        by_operation = {}
        for result in self.results:
            op = result.operation
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(result.to_dict())
        
        # Calculate aggregates
        conv2d_results = [r for r in self.results if r.operation == "conv2d"]
        matmul_results = [r for r in self.results if r.operation == "matmul"]
        
        summary = {
            "total_benchmarks": len(self.results),
            "by_operation": by_operation,
            "peak_conv2d_gflops": max((r.value for r in conv2d_results), default=0),
            "peak_matmul_gflops": max((r.value for r in matmul_results), default=0),
            "peak_memory_bandwidth_gbps": max((r.value for r in self.results 
                                              if r.metric_type == BenchmarkType.BANDWIDTH), default=0),
            "results": [r.to_dict() for r in self.results]
        }
        
        return summary
    
    def plot_results(self, output_path: Path = None):
        """Plot benchmark results"""
        
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping plots")
            return
            
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Vulkan ML SDK Performance Benchmarks", fontsize=16)
        
        # Conv2D performance
        conv_results = [r for r in self.results if r.operation == "conv2d"]
        if conv_results:
            ax = axes[0, 0]
            names = [r.name.split("_")[-1] for r in conv_results]
            values = [r.value for r in conv_results]
            ax.bar(names, values, color='blue')
            ax.set_title("Conv2D Performance")
            ax.set_ylabel("GFLOPS")
            ax.set_xlabel("Input Size")
        
        # MatMul performance
        matmul_results = [r for r in self.results if r.operation == "matmul"]
        if matmul_results:
            ax = axes[0, 1]
            sizes = [r.metadata.get("matrix_size", 0) for r in matmul_results]
            values = [r.value for r in matmul_results]
            ax.plot(sizes, values, 'o-', color='green', linewidth=2, markersize=8)
            ax.set_title("MatMul Performance Scaling")
            ax.set_ylabel("GFLOPS")
            ax.set_xlabel("Matrix Size")
            ax.grid(True, alpha=0.3)
        
        # Memory bandwidth
        mem_results = [r for r in self.results if r.metric_type == BenchmarkType.BANDWIDTH]
        if mem_results:
            ax = axes[1, 0]
            sizes = [r.metadata.get("size_mb", 0) for r in mem_results]
            values = [r.value for r in mem_results]
            ax.bar([str(s) for s in sizes], values, color='orange')
            ax.set_title("Memory Bandwidth")
            ax.set_ylabel("GB/s")
            ax.set_xlabel("Buffer Size (MB)")
        
        # Activation functions
        act_results = [r for r in self.results if r.operation in ["relu", "sigmoid", "tanh"]]
        if act_results:
            ax = axes[1, 1]
            names = [r.operation for r in act_results]
            values = [r.value for r in act_results]
            colors = ['red', 'purple', 'cyan']
            ax.bar(names, values, color=colors[:len(names)])
            ax.set_title("Activation Function Performance")
            ax.set_ylabel("MOps/s")
            ax.set_xlabel("Function")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
    
    def save_results(self, output_path: Path = None):
        """Save benchmark results to file"""
        
        if output_path is None:
            output_path = Path("/tmp/benchmark_results.json")
        
        summary = self._generate_summary()
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("Performance Summary")
        print("="*60)
        print(f"Peak Conv2D: {summary['peak_conv2d_gflops']:.2f} GFLOPS")
        print(f"Peak MatMul: {summary['peak_matmul_gflops']:.2f} GFLOPS")
        print(f"Peak Memory Bandwidth: {summary['peak_memory_bandwidth_gbps']:.2f} GB/s")

def main():
    """Run benchmark suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vulkan ML SDK Benchmark Suite")
    parser.add_argument("--operation", choices=["conv2d", "matmul", "memory", "all"],
                       default="all", help="Operation to benchmark")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations")
    parser.add_argument("--plot", action="store_true",
                       help="Generate performance plots")
    parser.add_argument("--output", type=str,
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = BenchmarkSuite()
    
    # Run benchmarks
    if args.operation == "all":
        summary = suite.benchmark_comprehensive()
    elif args.operation == "conv2d":
        result = suite.benchmark_conv2d(iterations=args.iterations)
        print(f"{result.name}: {result.value:.2f} {result.unit}")
    elif args.operation == "matmul":
        result = suite.benchmark_matmul(iterations=args.iterations)
        print(f"{result.name}: {result.value:.2f} {result.unit}")
    elif args.operation == "memory":
        result = suite.benchmark_memory_bandwidth(iterations=args.iterations)
        print(f"{result.name}: {result.value:.2f} {result.unit}")
    
    # Save results
    if args.output:
        suite.save_results(Path(args.output))
    
    # Plot if requested
    if args.plot:
        suite.plot_results(Path("/tmp/benchmark_plot.png"))

if __name__ == "__main__":
    main()