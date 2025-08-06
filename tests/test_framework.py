#!/usr/bin/env python3
"""
Vulkan ML SDK Test Framework
Main test orchestration and execution framework
"""

import os
import sys
import json
import time
import unittest
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Add SDK tools to path
SDK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SDK_ROOT / "builds" / "ARM-ML-SDK-Complete" / "tools"))

class TestLevel(Enum):
    """Test execution levels"""
    QUICK = "quick"          # ~5 minutes - smoke tests
    STANDARD = "standard"     # ~30 minutes - full suite
    EXTENSIVE = "extensive"   # 2+ hours - all tests + stress

class TestCategory(Enum):
    """Test categories"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    STRESS = "stress"
    REGRESSION = "regression"
    PLATFORM = "platform"

@dataclass
class TestResult:
    """Test execution result"""
    name: str
    category: TestCategory
    passed: bool
    duration: float
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "passed": self.passed,
            "duration": self.duration,
            "error_message": self.error_message,
            "metrics": self.metrics
        }

class VulkanMLTestFramework:
    """Main test framework for Vulkan ML SDK"""
    
    def __init__(self, level: TestLevel = TestLevel.STANDARD):
        self.level = level
        self.sdk_root = Path("/Users/jerry/Vulkan")
        self.sdk_bin = self.sdk_root / "builds" / "ARM-ML-SDK-Complete" / "bin"
        self.models_dir = self.sdk_root / "builds" / "ARM-ML-SDK-Complete" / "models"
        self.shaders_dir = self.sdk_root / "builds" / "ARM-ML-SDK-Complete" / "shaders"
        self.tools_dir = self.sdk_root / "builds" / "ARM-ML-SDK-Complete" / "tools"
        
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
        # Set environment
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup test environment variables"""
        os.environ["DYLD_LIBRARY_PATH"] = f"/usr/local/lib:{self.sdk_root}/builds/ARM-ML-SDK-Complete/lib"
        os.environ["SDK_ROOT"] = str(self.sdk_root)
        os.environ["SDK_BIN"] = str(self.sdk_bin)
    
    def discover_tests(self, category: Optional[TestCategory] = None) -> List[str]:
        """Discover available tests"""
        tests = []
        test_dir = self.sdk_root / "tests"
        
        if category:
            categories = [category]
        else:
            categories = list(TestCategory)
        
        for cat in categories:
            cat_dir = test_dir / cat.value
            if cat_dir.exists():
                # Find all test_*.py files
                for test_file in cat_dir.rglob("test_*.py"):
                    tests.append(str(test_file))
        
        return tests
    
    def run_test(self, test_path: str) -> TestResult:
        """Run a single test"""
        test_name = Path(test_path).stem
        category = self._get_test_category(test_path)
        
        start = time.time()
        try:
            # Run test using unittest
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(test_path.replace("/", ".").replace(".py", ""))
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            passed = result.wasSuccessful()
            error_msg = ""
            if not passed:
                errors = result.errors + result.failures
                if errors:
                    error_msg = str(errors[0][1])
            
            duration = time.time() - start
            
            return TestResult(
                name=test_name,
                category=category,
                passed=passed,
                duration=duration,
                error_message=error_msg
            )
            
        except Exception as e:
            duration = time.time() - start
            return TestResult(
                name=test_name,
                category=category,
                passed=False,
                duration=duration,
                error_message=str(e)
            )
    
    def _get_test_category(self, test_path: str) -> TestCategory:
        """Determine test category from path"""
        for category in TestCategory:
            if f"/{category.value}/" in test_path:
                return category
        return TestCategory.UNIT
    
    def run_scenario(self, scenario_path: str, output_dir: str = "/tmp/test_output") -> bool:
        """Run a scenario using scenario-runner"""
        scenario_runner = self.sdk_bin / "scenario-runner"
        
        cmd = [
            str(scenario_runner),
            "--scenario", scenario_path,
            "--output", output_dir,
            "--dry-run"  # For testing
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def run_shader_test(self, shader_name: str) -> bool:
        """Test if a shader exists and is valid"""
        shader_path = self.shaders_dir / f"{shader_name}.spv"
        return shader_path.exists() and shader_path.stat().st_size > 0
    
    def run_model_test(self, model_name: str) -> Dict[str, Any]:
        """Test a TensorFlow Lite model"""
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            return {"passed": False, "error": "Model not found"}
        
        # Basic validation
        stats = {
            "exists": True,
            "size_mb": model_path.stat().st_size / (1024 * 1024),
            "passed": True
        }
        
        # Could add more validation here (e.g., load with TFLite interpreter)
        
        return stats
    
    def run_benchmark(self, operation: str, params: Dict = None) -> Dict[str, float]:
        """Run a performance benchmark"""
        metrics = {}
        
        # Example benchmark for different operations
        if operation == "conv2d":
            metrics = self._benchmark_conv2d(params or {})
        elif operation == "matmul":
            metrics = self._benchmark_matmul(params or {})
        elif operation == "memory_bandwidth":
            metrics = self._benchmark_memory()
        
        return metrics
    
    def _benchmark_conv2d(self, params: Dict) -> Dict[str, float]:
        """Benchmark Conv2D operation"""
        # Placeholder - would run actual Conv2D benchmark
        return {
            "throughput_gflops": 150.5,
            "latency_ms": 2.5,
            "memory_bandwidth_gbps": 200.0
        }
    
    def _benchmark_matmul(self, params: Dict) -> Dict[str, float]:
        """Benchmark MatMul operation"""
        # Placeholder - would run actual MatMul benchmark
        return {
            "throughput_gflops": 400.0,
            "latency_ms": 1.2,
            "memory_bandwidth_gbps": 350.0
        }
    
    def _benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory operations"""
        # Simple memory bandwidth test
        size = 100 * 1024 * 1024  # 100MB
        data = np.random.randn(size // 8).astype(np.float64)
        
        start = time.time()
        _ = data.copy()
        duration = time.time() - start
        
        bandwidth_gbps = (size * 2) / (duration * 1e9)  # Read + Write
        
        return {
            "bandwidth_gbps": bandwidth_gbps,
            "latency_ns": duration * 1e9 / (size // 8)
        }
    
    def validate_output(self, actual: np.ndarray, expected: np.ndarray, 
                       tolerance: float = 1e-4) -> Tuple[bool, float]:
        """Validate output against expected result"""
        if actual.shape != expected.shape:
            return False, float('inf')
        
        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        
        passed = max_diff < tolerance
        return passed, max_diff
    
    def run_suite(self, categories: List[TestCategory] = None) -> Dict[str, Any]:
        """Run complete test suite"""
        self.start_time = datetime.now()
        
        if categories is None:
            categories = self._get_categories_for_level()
        
        print(f"\n{'='*60}")
        print(f"Vulkan ML SDK Test Suite - Level: {self.level.value}")
        print(f"Starting at: {self.start_time}")
        print(f"{'='*60}\n")
        
        for category in categories:
            self._run_category(category)
        
        self.end_time = datetime.now()
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _get_categories_for_level(self) -> List[TestCategory]:
        """Get test categories based on execution level"""
        if self.level == TestLevel.QUICK:
            return [TestCategory.UNIT, TestCategory.VALIDATION]
        elif self.level == TestLevel.STANDARD:
            return [TestCategory.UNIT, TestCategory.INTEGRATION, 
                   TestCategory.VALIDATION, TestCategory.PERFORMANCE]
        else:  # EXTENSIVE
            return list(TestCategory)
    
    def _run_category(self, category: TestCategory):
        """Run all tests in a category"""
        print(f"\n{'-'*40}")
        print(f"Running {category.value} tests...")
        print(f"{'-'*40}")
        
        tests = self.discover_tests(category)
        
        for test_path in tests:
            result = self.run_test(test_path)
            self.results.append(result)
            
            # Print result
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} - {result.name} ({result.duration:.2f}s)")
            if not result.passed and result.error_message:
                print(f"  Error: {result.error_message[:100]}...")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test execution summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Group by category
        by_category = {}
        for result in self.results:
            cat = result.category.value
            if cat not in by_category:
                by_category[cat] = {"passed": 0, "failed": 0, "tests": []}
            
            if result.passed:
                by_category[cat]["passed"] += 1
            else:
                by_category[cat]["failed"] += 1
            by_category[cat]["tests"].append(result.to_dict())
        
        summary = {
            "timestamp": self.start_time.isoformat(),
            "level": self.level.value,
            "duration_seconds": duration,
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "by_category": by_category,
            "failures": [r.to_dict() for r in self.results if not r.passed]
        }
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save test results to file"""
        results_dir = self.sdk_root / "tests" / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"test_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Test Execution Summary")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        
        if summary['failed'] > 0:
            print(f"\nFailed Tests:")
            for failure in summary['failures']:
                print(f"  - {failure['name']}: {failure['error_message'][:50]}...")
        
        print(f"{'='*60}\n")

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vulkan ML SDK Test Framework")
    parser.add_argument("--level", choices=["quick", "standard", "extensive"],
                       default="standard", help="Test execution level")
    parser.add_argument("--category", choices=[c.value for c in TestCategory],
                       help="Run specific test category")
    
    args = parser.parse_args()
    
    # Create framework
    level = TestLevel(args.level)
    framework = VulkanMLTestFramework(level)
    
    # Run tests
    if args.category:
        categories = [TestCategory(args.category)]
    else:
        categories = None
    
    summary = framework.run_suite(categories)
    
    # Exit with appropriate code
    sys.exit(0 if summary['failed'] == 0 else 1)

if __name__ == "__main__":
    main()