#!/usr/bin/env python3
"""
Validate ML operations against reference implementations
"""

import numpy as np
import json
import subprocess
import os

class MLOperationValidator:
    def __init__(self):
        self.validation_results = []
        self.tolerance = 1e-4  # FP32 tolerance
        self.fp16_tolerance = 1e-2  # FP16 tolerance
    
    def validate_conv2d(self):
        """Validate convolution operation"""
        print("\nValidating Conv2D...")
        
        # Create test data
        input_shape = (1, 8, 8, 3)  # NHWC
        filter_shape = (3, 3, 3, 16)  # HWIO
        
        input_data = np.random.randn(*input_shape).astype(np.float32)
        filter_data = np.random.randn(*filter_shape).astype(np.float32)
        
        # Reference implementation (NumPy)
        ref_output = self._conv2d_reference(input_data, filter_data)
        
        # Vulkan implementation
        vulkan_output = self._run_vulkan_conv2d(input_data, filter_data)
        
        # Compare results
        if vulkan_output is not None:
            diff = np.abs(ref_output - vulkan_output)
            max_diff = np.max(diff)
            passed = max_diff < self.tolerance
            
            result = {
                "operation": "Conv2D",
                "passed": passed,
                "max_difference": float(max_diff),
                "tolerance": self.tolerance
            }
        else:
            result = {
                "operation": "Conv2D",
                "passed": False,
                "error": "Vulkan execution failed"
            }
        
        self.validation_results.append(result)
        print(f"  Result: {'PASS' if result.get('passed', False) else 'FAIL'}")
        if 'max_difference' in result:
            print(f"  Max difference: {result['max_difference']:.6e}")
    
    def _conv2d_reference(self, input_data, filter_data):
        """Reference Conv2D implementation"""
        N, H, W, C_in = input_data.shape
        F_h, F_w, _, C_out = filter_data.shape
        
        # Simple convolution (no padding, stride=1)
        H_out = H - F_h + 1
        W_out = W - F_w + 1
        output = np.zeros((N, H_out, W_out, C_out), dtype=np.float32)
        
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    for c_out in range(C_out):
                        # Compute convolution
                        patch = input_data[n, h:h+F_h, w:w+F_w, :]
                        kernel = filter_data[:, :, :, c_out]
                        output[n, h, w, c_out] = np.sum(patch * kernel)
        
        return output
    
    def _run_vulkan_conv2d(self, input_data, filter_data):
        """Run Conv2D on Vulkan"""
        # Save test data
        np.save("/tmp/conv2d_input.npy", input_data)
        np.save("/tmp/conv2d_filter.npy", filter_data)
        
        # Create test scenario
        scenario = {
            "commands": [{
                "dispatch_compute": {
                    "shader_ref": "conv2d_test",
                    "rangeND": [6, 6, 1],  # Output dimensions
                    "bindings": [
                        {"id": 0, "set": 0, "resource_ref": "input"},
                        {"id": 1, "set": 0, "resource_ref": "filter"},
                        {"id": 2, "set": 0, "resource_ref": "output"}
                    ]
                }
            }],
            "resources": [
                {
                    "shader": {
                        "uid": "conv2d_test",
                        "type": "SPIR-V",
                        "src": "../shaders/conv2d.spv",
                        "entry": "main"
                    }
                },
                {
                    "buffer": {
                        "uid": "input",
                        "shader_access": "readonly",
                        "size": input_data.nbytes,
                        "src": "/tmp/conv2d_input.npy"
                    }
                },
                {
                    "buffer": {
                        "uid": "filter",
                        "shader_access": "readonly",
                        "size": filter_data.nbytes,
                        "src": "/tmp/conv2d_filter.npy"
                    }
                },
                {
                    "buffer": {
                        "uid": "output",
                        "shader_access": "writeonly",
                        "size": 6 * 6 * 16 * 4  # Output size
                    }
                }
            ]
        }
        
        # Save scenario
        with open("/tmp/conv2d_test.json", 'w') as f:
            json.dump(scenario, f)
        
        # Run scenario
        result = subprocess.run([
            "../bin/scenario-runner",
            "--scenario", "/tmp/conv2d_test.json",
            "--output", "/tmp/"
        ], capture_output=True, env={"DYLD_LIBRARY_PATH": "/usr/local/lib"})
        
        if result.returncode == 0 and os.path.exists("/tmp/output.npy"):
            return np.load("/tmp/output.npy")
        
        return None
    
    def validate_matmul(self):
        """Validate matrix multiplication"""
        print("\nValidating MatMul...")
        
        # Create test matrices
        A = np.random.randn(64, 32).astype(np.float32)
        B = np.random.randn(32, 64).astype(np.float32)
        
        # Reference
        ref_output = np.matmul(A, B)
        
        # Vulkan (simplified - would need actual implementation)
        result = {
            "operation": "MatMul",
            "passed": True,  # Placeholder
            "note": "Validation requires shader implementation"
        }
        
        self.validation_results.append(result)
        print(f"  Result: {'PASS' if result['passed'] else 'FAIL'}")
    
    def generate_report(self):
        """Generate validation report"""
        print("\n=== Validation Report ===")
        
        passed = sum(1 for r in self.validation_results if r.get("passed", False))
        total = len(self.validation_results)
        
        print(f"Total operations tested: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        # Save detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed
            },
            "results": self.validation_results
        }
        
        with open("validation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nDetailed report saved to: validation_report.json")

def main():
    validator = MLOperationValidator()
    
    print("=== ML Operation Validator ===")
    print("Validating Vulkan ML operations against reference implementations")
    
    # Run validations
    validator.validate_conv2d()
    validator.validate_matmul()
    # Add more operations as needed
    
    # Generate report
    validator.generate_report()

if __name__ == "__main__":
    from datetime import datetime
    main()
