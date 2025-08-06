#!/usr/bin/env python3
"""
Result Validation Framework for Vulkan ML SDK Tests
Provides numerical comparison, image validation, and performance regression detection
"""

import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import hashlib
import pickle
from enum import Enum

class ValidationMode(Enum):
    EXACT = "exact"
    NUMERICAL = "numerical"
    STATISTICAL = "statistical"
    VISUAL = "visual"
    PERFORMANCE = "performance"

@dataclass
class ValidationResult:
    passed: bool
    mode: ValidationMode
    message: str
    metrics: Dict[str, float]
    details: Optional[Dict[str, Any]] = None

class ResultValidator:
    """Comprehensive result validation for ML operations"""
    
    def __init__(self, tolerance_rtol: float = 1e-5, tolerance_atol: float = 1e-8):
        self.tolerance_rtol = tolerance_rtol
        self.tolerance_atol = tolerance_atol
        self.baseline_path = Path("/Users/jerry/Vulkan/tests/data/baselines")
        self.baseline_path.mkdir(parents=True, exist_ok=True)
        
        # Operation-specific tolerances
        self.op_tolerances = {
            "conv2d": {"rtol": 1e-4, "atol": 1e-6},
            "matmul": {"rtol": 1e-5, "atol": 1e-7},
            "softmax": {"rtol": 1e-6, "atol": 1e-9},
            "quantize": {"rtol": 0.01, "atol": 1},  # Integer quantization
            "dequantize": {"rtol": 0.01, "atol": 0.1}
        }
        
        # Performance thresholds
        self.perf_thresholds = {
            "latency_regression": 1.2,  # 20% regression threshold
            "memory_regression": 1.1,   # 10% memory regression
            "accuracy_threshold": 0.95  # 95% accuracy required
        }
    
    def validate_output(self, 
                       actual: np.ndarray, 
                       expected: np.ndarray,
                       operation: str = "unknown",
                       mode: ValidationMode = ValidationMode.NUMERICAL) -> ValidationResult:
        """Validate output against expected results"""
        
        if mode == ValidationMode.EXACT:
            return self._validate_exact(actual, expected)
        elif mode == ValidationMode.NUMERICAL:
            return self._validate_numerical(actual, expected, operation)
        elif mode == ValidationMode.STATISTICAL:
            return self._validate_statistical(actual, expected)
        elif mode == ValidationMode.VISUAL:
            return self._validate_visual(actual, expected)
        elif mode == ValidationMode.PERFORMANCE:
            return self._validate_performance(actual, expected)
        else:
            raise ValueError(f"Unknown validation mode: {mode}")
    
    def _validate_exact(self, actual: np.ndarray, expected: np.ndarray) -> ValidationResult:
        """Exact match validation"""
        if actual.shape != expected.shape:
            return ValidationResult(
                passed=False,
                mode=ValidationMode.EXACT,
                message=f"Shape mismatch: {actual.shape} vs {expected.shape}",
                metrics={}
            )
        
        matches = np.array_equal(actual, expected)
        
        return ValidationResult(
            passed=matches,
            mode=ValidationMode.EXACT,
            message="Exact match" if matches else "Values differ",
            metrics={
                "num_differences": np.sum(actual != expected) if not matches else 0
            }
        )
    
    def _validate_numerical(self, 
                           actual: np.ndarray, 
                           expected: np.ndarray,
                           operation: str) -> ValidationResult:
        """Numerical tolerance validation"""
        
        if actual.shape != expected.shape:
            return ValidationResult(
                passed=False,
                mode=ValidationMode.NUMERICAL,
                message=f"Shape mismatch: {actual.shape} vs {expected.shape}",
                metrics={}
            )
        
        # Get operation-specific tolerances
        tol = self.op_tolerances.get(operation, 
                                     {"rtol": self.tolerance_rtol, "atol": self.tolerance_atol})
        
        # Check for NaN and Inf
        if np.any(np.isnan(actual)) or np.any(np.isinf(actual)):
            return ValidationResult(
                passed=False,
                mode=ValidationMode.NUMERICAL,
                message="Output contains NaN or Inf values",
                metrics={
                    "num_nan": np.sum(np.isnan(actual)),
                    "num_inf": np.sum(np.isinf(actual))
                }
            )
        
        # Numerical comparison
        close = np.allclose(actual, expected, rtol=tol["rtol"], atol=tol["atol"])
        
        # Calculate error metrics
        abs_error = np.abs(actual - expected)
        rel_error = np.abs((actual - expected) / (expected + 1e-10))
        
        metrics = {
            "max_abs_error": float(np.max(abs_error)),
            "mean_abs_error": float(np.mean(abs_error)),
            "max_rel_error": float(np.max(rel_error)),
            "mean_rel_error": float(np.mean(rel_error)),
            "percentile_95_error": float(np.percentile(abs_error, 95))
        }
        
        # Detailed error analysis
        if not close:
            error_positions = np.where(abs_error > tol["atol"] + tol["rtol"] * np.abs(expected))
            num_errors = len(error_positions[0])
            
            details = {
                "num_errors": num_errors,
                "error_rate": num_errors / actual.size,
                "error_positions": error_positions if num_errors < 100 else "Too many to list"
            }
        else:
            details = None
        
        return ValidationResult(
            passed=close,
            mode=ValidationMode.NUMERICAL,
            message=f"Numerical validation {'passed' if close else 'failed'}",
            metrics=metrics,
            details=details
        )
    
    def _validate_statistical(self, 
                            actual: np.ndarray, 
                            expected: np.ndarray) -> ValidationResult:
        """Statistical validation for distributions"""
        
        # Calculate statistical metrics
        metrics = {
            "actual_mean": float(np.mean(actual)),
            "expected_mean": float(np.mean(expected)),
            "actual_std": float(np.std(actual)),
            "expected_std": float(np.std(expected)),
            "actual_min": float(np.min(actual)),
            "actual_max": float(np.max(actual)),
            "expected_min": float(np.min(expected)),
            "expected_max": float(np.max(expected))
        }
        
        # Kolmogorov-Smirnov test for distribution similarity
        from scipy import stats
        if actual.size == expected.size:
            ks_statistic, p_value = stats.ks_2samp(actual.flatten(), expected.flatten())
            metrics["ks_statistic"] = float(ks_statistic)
            metrics["p_value"] = float(p_value)
            
            # Pass if distributions are similar (p > 0.05)
            passed = p_value > 0.05
        else:
            # Compare distribution parameters
            mean_close = np.abs(metrics["actual_mean"] - metrics["expected_mean"]) < 0.1
            std_close = np.abs(metrics["actual_std"] - metrics["expected_std"]) < 0.1
            passed = mean_close and std_close
        
        return ValidationResult(
            passed=passed,
            mode=ValidationMode.STATISTICAL,
            message=f"Statistical validation {'passed' if passed else 'failed'}",
            metrics=metrics
        )
    
    def _validate_visual(self, 
                        actual: np.ndarray, 
                        expected: np.ndarray) -> ValidationResult:
        """Visual validation for image outputs"""
        
        # Ensure images are in same format
        if len(actual.shape) == 4:  # Batch dimension
            actual = actual[0]
            expected = expected[0]
        
        # Calculate visual metrics
        metrics = {}
        
        # PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((actual - expected) ** 2)
        if mse > 0:
            max_pixel = max(actual.max(), expected.max())
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            metrics["psnr"] = float(psnr)
        else:
            metrics["psnr"] = float('inf')
        
        # SSIM (Structural Similarity Index)
        ssim = self._calculate_ssim(actual, expected)
        metrics["ssim"] = float(ssim)
        
        # Color histogram comparison
        if len(actual.shape) == 3:  # Color image
            hist_distance = self._compare_histograms(actual, expected)
            metrics["histogram_distance"] = float(hist_distance)
        
        # Pass thresholds
        passed = metrics.get("psnr", 0) > 30 and metrics.get("ssim", 0) > 0.9
        
        return ValidationResult(
            passed=passed,
            mode=ValidationMode.VISUAL,
            message=f"Visual validation {'passed' if passed else 'failed'}",
            metrics=metrics
        )
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        # Simplified SSIM calculation
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    def _compare_histograms(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare color histograms"""
        # Calculate histograms for each channel
        distance = 0
        for channel in range(img1.shape[-1]):
            hist1, _ = np.histogram(img1[..., channel], bins=256, range=(0, 255))
            hist2, _ = np.histogram(img2[..., channel], bins=256, range=(0, 255))
            
            # Normalize histograms
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # Chi-square distance
            distance += np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))
        
        return distance / img1.shape[-1]
    
    def _validate_performance(self, 
                             actual_metrics: Dict[str, float],
                             baseline_metrics: Dict[str, float]) -> ValidationResult:
        """Validate performance metrics against baseline"""
        
        metrics = {}
        regressions = []
        
        # Compare latency
        if "latency_ms" in actual_metrics and "latency_ms" in baseline_metrics:
            latency_ratio = actual_metrics["latency_ms"] / baseline_metrics["latency_ms"]
            metrics["latency_ratio"] = latency_ratio
            
            if latency_ratio > self.perf_thresholds["latency_regression"]:
                regressions.append(f"Latency regression: {latency_ratio:.2f}x slower")
        
        # Compare memory usage
        if "memory_mb" in actual_metrics and "memory_mb" in baseline_metrics:
            memory_ratio = actual_metrics["memory_mb"] / baseline_metrics["memory_mb"]
            metrics["memory_ratio"] = memory_ratio
            
            if memory_ratio > self.perf_thresholds["memory_regression"]:
                regressions.append(f"Memory regression: {memory_ratio:.2f}x more")
        
        # Compare throughput
        if "throughput" in actual_metrics and "throughput" in baseline_metrics:
            throughput_ratio = actual_metrics["throughput"] / baseline_metrics["throughput"]
            metrics["throughput_ratio"] = throughput_ratio
            
            if throughput_ratio < 0.8:  # 20% throughput degradation
                regressions.append(f"Throughput regression: {throughput_ratio:.2f}x")
        
        passed = len(regressions) == 0
        
        return ValidationResult(
            passed=passed,
            mode=ValidationMode.PERFORMANCE,
            message="No performance regressions" if passed else "; ".join(regressions),
            metrics=metrics
        )
    
    def save_baseline(self, 
                     data: np.ndarray, 
                     operation: str, 
                     config: Dict[str, Any]) -> str:
        """Save baseline data for future comparison"""
        
        # Create unique identifier for this baseline
        config_str = json.dumps(config, sort_keys=True)
        baseline_id = hashlib.md5(f"{operation}_{config_str}".encode()).hexdigest()
        
        baseline_file = self.baseline_path / f"{operation}_{baseline_id}.pkl"
        
        baseline_data = {
            "operation": operation,
            "config": config,
            "data": data,
            "timestamp": time.time(),
            "shape": data.shape,
            "dtype": str(data.dtype)
        }
        
        with open(baseline_file, 'wb') as f:
            pickle.dump(baseline_data, f)
        
        return str(baseline_file)
    
    def load_baseline(self, operation: str, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load baseline data for comparison"""
        
        config_str = json.dumps(config, sort_keys=True)
        baseline_id = hashlib.md5(f"{operation}_{config_str}".encode()).hexdigest()
        
        baseline_file = self.baseline_path / f"{operation}_{baseline_id}.pkl"
        
        if not baseline_file.exists():
            return None
        
        try:
            with open(baseline_file, 'rb') as f:
                baseline_data = pickle.load(f)
            return baseline_data["data"]
        except Exception as e:
            print(f"Error loading baseline: {e}")
            return None
    
    def validate_model_output(self, 
                             model_name: str,
                             actual_output: np.ndarray,
                             test_input: np.ndarray) -> ValidationResult:
        """Validate complete model output"""
        
        # Model-specific validation logic
        if "mobilenet" in model_name.lower():
            return self._validate_classification_output(actual_output)
        elif any(style in model_name.lower() for style in ["la_muse", "udnie", "wave", "mirror"]):
            return self._validate_style_transfer_output(actual_output)
        elif "fire" in model_name.lower():
            return self._validate_detection_output(actual_output)
        else:
            # Generic validation
            return self._validate_generic_model_output(actual_output)
    
    def _validate_classification_output(self, output: np.ndarray) -> ValidationResult:
        """Validate classification model output"""
        
        # Check output shape (should be probabilities)
        if len(output.shape) != 2 or output.shape[0] != 1:
            return ValidationResult(
                passed=False,
                mode=ValidationMode.STATISTICAL,
                message=f"Invalid classification output shape: {output.shape}",
                metrics={}
            )
        
        # Check probability constraints
        prob_sum = np.sum(output)
        valid_probs = np.all(output >= 0) and np.all(output <= 1)
        
        metrics = {
            "num_classes": output.shape[1],
            "prob_sum": float(prob_sum),
            "max_prob": float(np.max(output)),
            "min_prob": float(np.min(output)),
            "top_class": int(np.argmax(output))
        }
        
        passed = valid_probs and np.abs(prob_sum - 1.0) < 0.01
        
        return ValidationResult(
            passed=passed,
            mode=ValidationMode.STATISTICAL,
            message="Classification output valid" if passed else "Invalid probabilities",
            metrics=metrics
        )
    
    def _validate_style_transfer_output(self, output: np.ndarray) -> ValidationResult:
        """Validate style transfer model output"""
        
        # Check output is valid image
        if len(output.shape) != 4 or output.shape[-1] != 3:
            return ValidationResult(
                passed=False,
                mode=ValidationMode.VISUAL,
                message=f"Invalid image output shape: {output.shape}",
                metrics={}
            )
        
        # Check pixel value range
        min_val = np.min(output)
        max_val = np.max(output)
        
        metrics = {
            "min_pixel": float(min_val),
            "max_pixel": float(max_val),
            "mean_pixel": float(np.mean(output)),
            "std_pixel": float(np.std(output))
        }
        
        # Style transfer outputs should be in reasonable range
        passed = min_val >= -10 and max_val <= 265  # Some margin beyond [0, 255]
        
        return ValidationResult(
            passed=passed,
            mode=ValidationMode.VISUAL,
            message="Style transfer output valid" if passed else "Pixel values out of range",
            metrics=metrics
        )
    
    def _validate_detection_output(self, output: np.ndarray) -> ValidationResult:
        """Validate detection model output"""
        
        # Detection outputs typically have boxes and scores
        metrics = {
            "output_shape": str(output.shape),
            "num_detections": output.shape[0] if len(output.shape) > 0 else 0
        }
        
        # Basic shape validation
        passed = len(output.shape) >= 2
        
        return ValidationResult(
            passed=passed,
            mode=ValidationMode.STATISTICAL,
            message="Detection output valid" if passed else "Invalid detection format",
            metrics=metrics
        )
    
    def _validate_generic_model_output(self, output: np.ndarray) -> ValidationResult:
        """Generic model output validation"""
        
        # Check for NaN/Inf
        has_nan = np.any(np.isnan(output))
        has_inf = np.any(np.isinf(output))
        
        metrics = {
            "shape": str(output.shape),
            "dtype": str(output.dtype),
            "min": float(np.min(output)) if not has_nan else "NaN",
            "max": float(np.max(output)) if not has_nan else "NaN",
            "mean": float(np.mean(output)) if not has_nan else "NaN"
        }
        
        passed = not has_nan and not has_inf
        
        return ValidationResult(
            passed=passed,
            mode=ValidationMode.NUMERICAL,
            message="Output valid" if passed else "Output contains NaN/Inf",
            metrics=metrics
        )
    
    def compare_implementations(self,
                               vulkan_output: np.ndarray,
                               reference_output: np.ndarray,
                               operation: str) -> Dict[str, Any]:
        """Compare Vulkan implementation with reference (NumPy/TensorFlow)"""
        
        results = {}
        
        # Numerical validation
        numerical_result = self.validate_output(
            vulkan_output, reference_output, operation, ValidationMode.NUMERICAL
        )
        results["numerical"] = {
            "passed": numerical_result.passed,
            "metrics": numerical_result.metrics
        }
        
        # Statistical validation
        statistical_result = self.validate_output(
            vulkan_output, reference_output, operation, ValidationMode.STATISTICAL
        )
        results["statistical"] = {
            "passed": statistical_result.passed,
            "metrics": statistical_result.metrics
        }
        
        # Performance comparison (if metrics available)
        if hasattr(vulkan_output, 'execution_time'):
            results["performance"] = {
                "vulkan_time_ms": vulkan_output.execution_time,
                "speedup": reference_output.execution_time / vulkan_output.execution_time
            }
        
        # Overall verdict
        results["overall_passed"] = all(
            r.get("passed", False) for r in results.values() if isinstance(r, dict)
        )
        
        return results

if __name__ == "__main__":
    # Example usage
    validator = ResultValidator()
    
    # Test numerical validation
    actual = np.random.randn(10, 10).astype(np.float32)
    expected = actual + np.random.randn(10, 10).astype(np.float32) * 1e-6
    
    result = validator.validate_output(actual, expected, "matmul")
    print(f"Validation result: {result.passed}")
    print(f"Metrics: {result.metrics}")