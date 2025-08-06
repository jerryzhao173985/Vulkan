#!/usr/bin/env python3
"""
Vulkan ML SDK Test Framework
Main test orchestration and execution engine
"""

import os
import sys
import json
import time
import glob
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import platform

class TestLevel(Enum):
    QUICK = "quick"        # 5 min - basic smoke tests
    STANDARD = "standard"  # 30 min - full test suite
    EXTENSIVE = "extensive" # 2+ hrs - all tests + stress

class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class TestResult:
    name: str
    category: str
    status: TestStatus
    duration: float
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None

@dataclass
class PerformanceMetrics:
    memory_usage_mb: float
    cpu_percent: float
    gpu_utilization: Optional[float]
    inference_time_ms: float
    throughput_ops_sec: float
    bandwidth_gb_s: Optional[float]

class VulkanMLTestFramework:
    """Main test orchestration framework for Vulkan ML SDK"""
    
    def __init__(self, sdk_path: str = "/Users/jerry/Vulkan/builds/ARM-ML-SDK-Complete"):
        self.sdk_path = Path(sdk_path)
        self.test_root = Path("/Users/jerry/Vulkan/tests")
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        self.system_info = self._gather_system_info()
        
        # Test categories and their weights
        self.test_categories = {
            "unit": {"weight": 1.0, "timeout": 60},
            "integration": {"weight": 2.0, "timeout": 180},
            "performance": {"weight": 1.5, "timeout": 300},
            "validation": {"weight": 2.0, "timeout": 120},
            "stress": {"weight": 1.0, "timeout": 600},
            "regression": {"weight": 1.5, "timeout": 120},
            "platform": {"weight": 1.0, "timeout": 90}
        }
        
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information for test context"""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.machine(),
            "cpu_count": mp.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3) if HAS_PSUTIL else "N/A",
            "python_version": sys.version,
            "sdk_version": self._get_sdk_version(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_sdk_version(self) -> str:
        """Get SDK version from scenario-runner"""
        try:
            cmd = f"{self.sdk_path}/bin/scenario-runner --version"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "version" in line.lower():
                        return line.strip()
            return "unknown"
        except:
            return "unknown"
    
    def discover_tests(self, level: TestLevel = TestLevel.STANDARD) -> List[Dict[str, Any]]:
        """Discover all available tests based on level"""
        tests = []
        
        # Define test patterns based on level
        patterns = {
            TestLevel.QUICK: ["**/test_basic*.py", "**/test_smoke*.py"],
            TestLevel.STANDARD: ["**/test_*.py", "**/test_*.cpp"],
            TestLevel.EXTENSIVE: ["**/*test*.py", "**/*test*.cpp", "**/benchmark*.py"]
        }
        
        for category in self.test_categories.keys():
            category_path = self.test_root / category
            if not category_path.exists():
                continue
                
            for pattern in patterns[level]:
                for test_file in category_path.glob(pattern):
                    tests.append({
                        "file": str(test_file),
                        "category": category,
                        "name": test_file.stem,
                        "timeout": self.test_categories[category]["timeout"]
                    })
        
        return tests
    
    def run_test(self, test_info: Dict[str, Any]) -> TestResult:
        """Execute a single test and return results"""
        start_time = time.time()
        test_file = test_info["file"]
        
        try:
            # Determine test runner based on file extension
            if test_file.endswith(".py"):
                cmd = f"python3 {test_file}"
            elif test_file.endswith(".cpp"):
                # Compile and run C++ test
                executable = test_file.replace(".cpp", "")
                compile_cmd = f"c++ -std=c++17 -O2 {test_file} -o {executable} -lpthread"
                subprocess.run(compile_cmd, shell=True, check=True, timeout=30)
                cmd = executable
            else:
                cmd = f"bash {test_file}"
            
            # Run test with timeout
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=test_info["timeout"],
                env={**os.environ, "SDK_PATH": str(self.sdk_path)}
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                status = TestStatus.PASSED
                message = "Test completed successfully"
            else:
                status = TestStatus.FAILED
                message = f"Test failed with exit code {result.returncode}"
            
            # Parse test output for metrics
            metrics = self._parse_test_output(result.stdout)
            
            return TestResult(
                name=test_info["name"],
                category=test_info["category"],
                status=status,
                duration=duration,
                message=message,
                metrics=metrics,
                error_details=result.stderr if status == TestStatus.FAILED else None
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                name=test_info["name"],
                category=test_info["category"],
                status=TestStatus.TIMEOUT,
                duration=test_info["timeout"],
                message=f"Test timed out after {test_info['timeout']}s"
            )
        except Exception as e:
            return TestResult(
                name=test_info["name"],
                category=test_info["category"],
                status=TestStatus.ERROR,
                duration=time.time() - start_time,
                message=f"Test error: {str(e)}",
                error_details=str(e)
            )
    
    def _parse_test_output(self, output: str) -> Dict[str, Any]:
        """Parse test output for metrics"""
        metrics = {}
        
        # Parse common metric patterns
        patterns = {
            "throughput": r"throughput[:\s]+([0-9.]+)\s*(ops/s|GFLOPS)",
            "latency": r"latency[:\s]+([0-9.]+)\s*ms",
            "accuracy": r"accuracy[:\s]+([0-9.]+)%?",
            "memory": r"memory[:\s]+([0-9.]+)\s*(MB|GB)",
            "bandwidth": r"bandwidth[:\s]+([0-9.]+)\s*GB/s"
        }
        
        import re
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))
        
        return metrics
    
    def run_suite(self, level: TestLevel = TestLevel.STANDARD, 
                  parallel: bool = True, max_workers: int = 4) -> None:
        """Run complete test suite"""
        self.start_time = time.time()
        tests = self.discover_tests(level)
        
        print(f"╔{'═' * 60}╗")
        print(f"║{'Vulkan ML SDK Test Suite':^60}║")
        print(f"║{'Level: ' + level.value:^60}║")
        print(f"║{'Tests: ' + str(len(tests)):^60}║")
        print(f"╚{'═' * 60}╝\n")
        
        if parallel and len(tests) > 1:
            with mp.Pool(processes=min(max_workers, mp.cpu_count())) as pool:
                self.results = pool.map(self.run_test, tests)
        else:
            self.results = [self.run_test(test) for test in tests]
        
        self.end_time = time.time()
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print test execution summary"""
        total_duration = self.end_time - self.start_time
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        timeouts = sum(1 for r in self.results if r.status == TestStatus.TIMEOUT)
        
        print(f"\n{'=' * 60}")
        print(f"Test Suite Summary")
        print(f"{'=' * 60}")
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed} ({100*passed/len(self.results):.1f}%)")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Timeouts: {timeouts}")
        print(f"Duration: {total_duration:.2f}s")
        
        # Print failed tests
        if failed > 0:
            print(f"\nFailed Tests:")
            for result in self.results:
                if result.status == TestStatus.FAILED:
                    print(f"  - {result.category}/{result.name}: {result.message}")
        
        # Calculate test score
        score = self._calculate_score()
        print(f"\nTest Score: {score:.1f}/100")
        
        if score >= 95:
            print("✅ Excellent - Production Ready")
        elif score >= 80:
            print("✅ Good - Minor Issues")
        elif score >= 60:
            print("⚠️  Fair - Needs Improvement")
        else:
            print("❌ Poor - Major Issues")
    
    def _calculate_score(self) -> float:
        """Calculate weighted test score"""
        if not self.results:
            return 0.0
        
        category_scores = {}
        for category in self.test_categories:
            category_results = [r for r in self.results if r.category == category]
            if category_results:
                passed = sum(1 for r in category_results if r.status == TestStatus.PASSED)
                category_scores[category] = 100 * passed / len(category_results)
        
        # Calculate weighted average
        total_weight = sum(self.test_categories[c]["weight"] for c in category_scores)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            category_scores[c] * self.test_categories[c]["weight"] 
            for c in category_scores
        )
        
        return weighted_sum / total_weight
    
    def generate_report(self, output_path: str = "test_report.json") -> None:
        """Generate detailed test report"""
        report = {
            "system_info": self.system_info,
            "execution": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": self.end_time - self.start_time if self.end_time else 0
            },
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == TestStatus.PASSED),
                "failed": sum(1 for r in self.results if r.status == TestStatus.FAILED),
                "errors": sum(1 for r in self.results if r.status == TestStatus.ERROR),
                "timeouts": sum(1 for r in self.results if r.status == TestStatus.TIMEOUT),
                "score": self._calculate_score()
            },
            "results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "status": r.status.value,
                    "duration": r.duration,
                    "message": r.message,
                    "metrics": r.metrics,
                    "error_details": r.error_details
                }
                for r in self.results
            ]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nReport saved to: {output_file}")
    
    def run_continuous(self, interval_minutes: int = 30) -> None:
        """Run continuous testing at specified intervals"""
        print(f"Starting continuous testing (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                self.run_suite(TestLevel.QUICK)
                self.generate_report(f"reports/continuous_{datetime.now():%Y%m%d_%H%M%S}.json")
                
                # Check for regressions
                if self._detect_regressions():
                    print("⚠️  Performance regression detected!")
                
                print(f"Next run in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\nContinuous testing stopped")
                break
    
    def _detect_regressions(self) -> bool:
        """Detect performance regressions from previous runs"""
        # Compare with baseline metrics
        # This would load previous results and compare
        return False  # Placeholder

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vulkan ML SDK Test Framework")
    parser.add_argument("--level", choices=["quick", "standard", "extensive"], 
                       default="standard", help="Test level to run")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--continuous", action="store_true", help="Run continuous testing")
    parser.add_argument("--report", default="test_report.json", help="Report output path")
    
    args = parser.parse_args()
    
    framework = VulkanMLTestFramework()
    
    if args.continuous:
        framework.run_continuous()
    else:
        level = TestLevel(args.level)
        framework.run_suite(level, args.parallel, args.workers)
        framework.generate_report(args.report)