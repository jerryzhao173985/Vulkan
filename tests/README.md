# Vulkan ML SDK - Comprehensive Test Suite

## Overview

This is a comprehensive, multi-layered test suite for the ARM ML SDK for Vulkan on macOS ARM64. The test suite validates correctness, performance, and reliability across all SDK components.

## Test Architecture

```
tests/
├── unit/                    # Component-level unit tests
│   ├── vgf/                # VGF library tests
│   ├── converter/           # Model converter tests
│   ├── runner/              # Scenario runner tests
│   └── emulation/           # Emulation layer tests
│
├── integration/             # Cross-component integration tests
│   ├── model_pipeline/      # End-to-end model conversion
│   ├── inference/           # Complete inference paths
│   └── shader_execution/    # Shader integration tests
│
├── performance/             # Performance benchmarks
│   ├── operations/          # Individual ML op benchmarks
│   ├── models/              # Full model benchmarks
│   └── memory/              # Memory/bandwidth tests
│
├── validation/              # Correctness validation
│   ├── numerical/           # Numerical accuracy tests
│   ├── ml_operations/       # ML op validation
│   └── outputs/             # Output comparison tests
│
├── stress/                  # Stress & edge case tests
│   ├── large_models/        # Large model handling
│   ├── concurrent/          # Concurrent execution
│   └── resource_limits/     # Resource exhaustion tests
│
├── regression/              # Regression test suite
│   ├── known_issues/        # Tests for fixed bugs
│   └── compatibility/       # Version compatibility
│
├── platform/                # Platform-specific tests
│   └── macos_arm64/         # Apple Silicon optimizations
│
├── test_framework.py        # Main test orchestration framework
├── test_scenarios.py        # Test scenario generator
├── test_validation.py       # Result validation utilities
├── test_benchmarks.py       # Performance benchmark suite
├── run_test_suite.sh        # Master test runner script
└── run_all_tests.sh         # Legacy compatibility script
```

## Quick Start

### Run Quick Tests (5 minutes)
```bash
./tests/run_test_suite.sh --level quick
```

### Run Standard Tests (30 minutes)
```bash
./tests/run_test_suite.sh --level standard
```

### Run Extensive Tests (2+ hours)
```bash
./tests/run_test_suite.sh --level extensive
```

### Run Specific Category
```bash
./tests/run_test_suite.sh --category validation
```

## Test Components

### 1. Test Framework (`test_framework.py`)

The main test orchestration framework provides:
- Test discovery and execution
- Result aggregation and reporting
- Performance metric collection
- Regression detection

**Key Classes:**
- `VulkanMLTestFramework`: Main test framework
- `TestResult`: Test execution result container
- `TestLevel`: Execution level enumeration (QUICK, STANDARD, EXTENSIVE)
- `TestCategory`: Test category enumeration

**Usage:**
```python
from test_framework import VulkanMLTestFramework, TestLevel

framework = VulkanMLTestFramework(TestLevel.STANDARD)
results = framework.run_suite()
```

### 2. Scenario Generator (`test_scenarios.py`)

Generates test scenarios for various ML operations and edge cases.

**Key Classes:**
- `ScenarioGenerator`: Main scenario generation class
- `OperationType`: ML operation types (Conv2D, MatMul, etc.)

**Features:**
- Operation-specific scenario generation
- Edge case scenarios
- Benchmark scenarios
- Model inference scenarios

**Usage:**
```python
from test_scenarios import ScenarioGenerator

generator = ScenarioGenerator()
scenarios = generator.generate_all_scenarios()
```

### 3. Validation Suite (`test_validation.py`)

Validates outputs against reference implementations.

**Key Classes:**
- `ResultValidator`: Main validation class
- `MLOperationValidator`: ML operation-specific validation
- `ValidationMode`: Comparison modes (EXACT, ABSOLUTE, RELATIVE, STATISTICAL)

**Features:**
- Numerical accuracy validation
- Reference implementation comparison
- Statistical similarity checks
- Tolerance-based validation

**Usage:**
```python
from test_validation import ResultValidator, ValidationMode

validator = ResultValidator()
result = validator.validate_tensor(actual, expected, ValidationMode.ABSOLUTE)
```

### 4. Benchmark Suite (`test_benchmarks.py`)

Comprehensive performance benchmarking for ML operations.

**Key Classes:**
- `BenchmarkSuite`: Main benchmark suite
- `BenchmarkResult`: Benchmark result container
- `BenchmarkType`: Metric types (LATENCY, THROUGHPUT, BANDWIDTH)

**Features:**
- Operation benchmarks (Conv2D, MatMul, activations)
- Memory bandwidth testing
- Model inference benchmarks
- Performance scaling analysis

**Usage:**
```python
from test_benchmarks import BenchmarkSuite

suite = BenchmarkSuite()
results = suite.benchmark_comprehensive()
suite.plot_results()
```

## Test Levels

### Quick (5 minutes)
- Basic smoke tests
- Binary existence checks
- Library loading tests
- Quick validation tests

### Standard (30 minutes)
- All quick tests
- Unit tests for each component
- Integration tests
- Performance benchmarks
- Validation tests

### Extensive (2+ hours)
- All standard tests
- Stress tests with large models
- Concurrent execution tests
- Resource limit tests
- Regression tests
- Extended benchmarks

## Test Categories

### Unit Tests
Test individual components in isolation:
- VGF encoder/decoder
- Model converter operations
- Scenario runner parsing
- Emulation layer functions

### Integration Tests
Test component interactions:
- Model conversion pipeline
- End-to-end inference
- Shader compilation and execution
- Resource management

### Performance Tests
Measure performance metrics:
- Operation throughput (GFLOPS)
- Memory bandwidth (GB/s)
- Inference latency (ms)
- Power efficiency (ops/watt)

### Validation Tests
Verify correctness:
- Numerical accuracy (FP32, FP16)
- Operation validation against references
- Output comparison
- Statistical similarity

### Stress Tests
Test under extreme conditions:
- Large models (>100MB)
- Concurrent execution
- Memory pressure
- Resource exhaustion

### Regression Tests
Prevent regressions:
- Known bug fixes
- Compatibility tests
- API stability

## Running Tests

### Command Line Options

```bash
./tests/run_test_suite.sh [OPTIONS]

Options:
  --level <quick|standard|extensive>  Test execution level
  --category <category>                Run specific category
  --verbose                           Verbose output
  --no-report                         Skip report generation
  --benchmarks                        Include performance benchmarks
  --help                              Show help message
```

### Python Framework

```python
# Run with Python framework
python3 tests/test_framework.py --level standard --category validation
```

### Individual Tests

```python
# Run specific test module
python3 tests/test_validation.py

# Run benchmark suite
python3 tests/test_benchmarks.py --operation all --iterations 100 --plot
```

## Test Reports

### JSON Report
Contains detailed test results in machine-readable format:
- Test execution times
- Pass/fail status
- Performance metrics
- Error messages

### HTML Report
User-friendly report with:
- Summary statistics
- Charts and graphs
- Test categories breakdown
- Performance visualizations

### Log Files
Complete execution logs saved to:
```
tests/results/test_log_YYYYMMDD_HHMMSS.log
```

## Performance Metrics

### Expected Performance (M4 Max)

| Operation | Expected Performance |
|-----------|---------------------|
| Conv2D (224x224x3) | ~150 GFLOPS |
| MatMul (1024x1024) | ~400 GFLOPS |
| Memory Bandwidth | ~400 GB/s |
| ReLU Activation | ~10 GOps/s |
| Model Inference (MobileNet) | ~15ms |
| Style Transfer | ~150ms |

## Adding New Tests

### 1. Create Test File

Place in appropriate category directory:
```python
# tests/unit/vgf/test_new_feature.py
import unittest
from test_framework import VulkanMLTestFramework

class TestNewFeature(unittest.TestCase):
    def test_feature(self):
        # Test implementation
        self.assertTrue(result)
```

### 2. Add to Scenario Generator

For new operations:
```python
# In test_scenarios.py
def _add_new_operation_resources(self, scenario, params):
    # Add operation-specific resources
    return scenario
```

### 3. Add Validation

For new validation requirements:
```python
# In test_validation.py
def validate_new_operation(self, actual, expected):
    # Validation logic
    return self.validate_tensor(actual, expected)
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Tests
        run: ./tests/run_test_suite.sh --level quick
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: tests/results/
```

## Troubleshooting

### Common Issues

1. **Library not found**
   ```bash
   export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK_ROOT/builds/ARM-ML-SDK-Complete/lib
   ```

2. **Python module not found**
   ```bash
   export PYTHONPATH=$SDK_ROOT/tests:$PYTHONPATH
   ```

3. **Scenario runner not found**
   ```bash
   # Ensure SDK is built
   ./scripts/build/build_optimized.sh
   ```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Reproducibility**: Use fixed seeds for random data
3. **Performance**: Cache test data when possible
4. **Coverage**: Aim for >90% code coverage
5. **Documentation**: Document test purpose and expected behavior

## Contributing

1. Follow test naming conventions: `test_<feature>_<aspect>.py`
2. Add appropriate assertions and error messages
3. Include performance baselines for benchmarks
4. Update documentation for new test categories

## License

Tests are part of the ARM ML SDK for Vulkan project.

---

*Last Updated: 2025-08-05*
*Platform: macOS ARM64 (Apple Silicon)*