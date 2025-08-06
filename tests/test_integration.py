#!/usr/bin/env python3
"""
Integration test to verify SDK functionality
"""

import os
import sys
import subprocess
from pathlib import Path

def test_sdk_components():
    """Test that all SDK components are present and functional"""
    sdk_root = Path("/Users/jerry/Vulkan")
    sdk_dir = sdk_root / "builds" / "ARM-ML-SDK-Complete"
    
    print("=" * 50)
    print("SDK Component Integration Test")
    print("=" * 50)
    
    # Test 1: Check binary exists and runs
    print("\n1. Testing scenario-runner binary...")
    scenario_runner = sdk_dir / "bin" / "scenario-runner"
    if scenario_runner.exists():
        try:
            result = subprocess.run([str(scenario_runner), "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   ✓ scenario-runner works")
            else:
                print(f"   ✗ scenario-runner failed: {result.stderr[:100]}")
        except Exception as e:
            print(f"   ✗ Error running scenario-runner: {e}")
    else:
        print("   ✗ scenario-runner not found")
    
    # Test 2: Check libraries
    print("\n2. Testing libraries...")
    lib_dir = sdk_dir / "lib"
    required_libs = ["libvgf.a", "libSPIRV.a", "libSPIRV-Tools.a"]
    for lib in required_libs:
        lib_path = lib_dir / lib
        if lib_path.exists():
            size_mb = lib_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {lib} ({size_mb:.2f} MB)")
        else:
            print(f"   ✗ {lib} not found")
    
    # Test 3: Check models
    print("\n3. Testing ML models...")
    models_dir = sdk_dir / "models"
    model_count = len(list(models_dir.glob("*.tflite")))
    if model_count > 0:
        print(f"   ✓ Found {model_count} TFLite models")
        for model in list(models_dir.glob("*.tflite"))[:3]:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"      - {model.name} ({size_mb:.2f} MB)")
    else:
        print("   ✗ No models found")
    
    # Test 4: Check shaders
    print("\n4. Testing shaders...")
    shaders_dir = sdk_dir / "shaders"
    shader_count = len(list(shaders_dir.glob("*.spv")))
    if shader_count > 0:
        print(f"   ✓ Found {shader_count} compiled shaders")
        important_shaders = ["add.spv", "multiply.spv", "conv2d.spv", "matmul.spv"]
        for shader_name in important_shaders:
            shader_path = shaders_dir / shader_name
            if shader_path.exists():
                print(f"      ✓ {shader_name}")
            else:
                # Check for variations
                similar = list(shaders_dir.glob(f"*{shader_name.split('.')[0]}*.spv"))
                if similar:
                    print(f"      ✓ {similar[0].name} (variant)")
    else:
        print("   ✗ No shaders found")
    
    # Test 5: Python tools
    print("\n5. Testing Python tools...")
    tools_dir = sdk_dir / "tools"
    if tools_dir.exists():
        py_tools = list(tools_dir.glob("*.py"))
        print(f"   ✓ Found {len(py_tools)} Python tools")
        for tool in py_tools[:3]:
            print(f"      - {tool.name}")
    else:
        print("   ✗ Tools directory not found")
    
    # Test 6: Basic NumPy operations
    print("\n6. Testing NumPy operations...")
    try:
        import numpy as np
        
        # Simple computation test
        a = np.random.randn(100, 100).astype(np.float32)
        b = np.random.randn(100, 100).astype(np.float32)
        c = np.matmul(a, b)
        
        # Verify result
        expected_shape = (100, 100)
        if c.shape == expected_shape:
            print(f"   ✓ NumPy matmul works: {c.shape}")
        else:
            print(f"   ✗ NumPy matmul failed: {c.shape}")
        
        # Memory bandwidth test
        size_mb = 10
        data = np.random.randn(size_mb * 1024 * 1024 // 8).astype(np.float64)
        import time
        start = time.time()
        copy = data.copy()
        elapsed = time.time() - start
        bandwidth_gbps = (size_mb * 2 * 1024 * 1024) / (elapsed * 1e9)
        print(f"   ✓ Memory bandwidth: {bandwidth_gbps:.2f} GB/s")
        
    except ImportError:
        print("   ✗ NumPy not available")
    except Exception as e:
        print(f"   ✗ NumPy test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Integration Test Complete")
    print("=" * 50)

if __name__ == "__main__":
    test_sdk_components()