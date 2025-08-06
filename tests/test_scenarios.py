#!/usr/bin/env python3
"""
Test Scenario Generator for Vulkan ML SDK
Generates test scenarios for various ML operations and edge cases
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

class OperationType(Enum):
    """ML Operation types"""
    CONV2D = "conv2d"
    MATMUL = "matmul"
    MAXPOOL = "maxpool"
    AVGPOOL = "avgpool"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    BATCHNORM = "batchnorm"
    ADD = "add"
    MUL = "multiply"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    CONCAT = "concat"
    SLICE = "slice"

class DataFormat(Enum):
    """Tensor data formats"""
    NHWC = "nhwc"  # Batch, Height, Width, Channels
    NCHW = "nchw"  # Batch, Channels, Height, Width
    NC = "nc"      # Batch, Channels (for fully connected)

class ScenarioGenerator:
    """Generate test scenarios for ML operations"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("/tmp/test_scenarios")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Shader mappings
        self.shader_map = {
            OperationType.ADD: "add",
            OperationType.MUL: "multiply",
            OperationType.RELU: "relu",
            OperationType.SIGMOID: "sigmoid",
            OperationType.CONV2D: "optimized_conv2d",
            OperationType.MATMUL: "matrix_multiply",
            OperationType.MAXPOOL: "maxpool2d",
            OperationType.AVGPOOL: "avgpool2d"
        }
    
    def generate_operation_scenario(self, 
                                   op_type: OperationType,
                                   input_shape: Tuple[int, ...],
                                   params: Dict[str, Any] = None) -> Dict:
        """Generate a scenario for a specific operation"""
        
        scenario = {
            "name": f"test_{op_type.value}",
            "description": f"Test scenario for {op_type.value} operation",
            "resources": [],
            "commands": []
        }
        
        # Generate input resources
        input_resource = self._create_tensor_resource(
            uid="_input",
            dims=list(input_shape),
            format="VK_FORMAT_R32_SFLOAT",
            access="readonly"
        )
        scenario["resources"].append(input_resource)
        
        # Generate operation-specific resources and commands
        if op_type == OperationType.CONV2D:
            scenario = self._add_conv2d_resources(scenario, input_shape, params)
        elif op_type == OperationType.MATMUL:
            scenario = self._add_matmul_resources(scenario, input_shape, params)
        elif op_type in [OperationType.RELU, OperationType.SIGMOID, OperationType.TANH]:
            scenario = self._add_activation_resources(scenario, op_type, input_shape)
        elif op_type in [OperationType.MAXPOOL, OperationType.AVGPOOL]:
            scenario = self._add_pooling_resources(scenario, op_type, input_shape, params)
        elif op_type in [OperationType.ADD, OperationType.MUL]:
            scenario = self._add_binary_op_resources(scenario, op_type, input_shape)
        
        return scenario
    
    def _create_tensor_resource(self, uid: str, dims: List[int], 
                               format: str, access: str,
                               src: str = None, dst: str = None) -> Dict:
        """Create a tensor resource definition"""
        resource = {
            "tensor": {
                "uid": uid,
                "shader_access": access,
                "dims": dims,
                "format": format
            }
        }
        
        if src:
            resource["tensor"]["src"] = src
        if dst:
            resource["tensor"]["dst"] = dst
            
        return resource
    
    def _create_buffer_resource(self, uid: str, size: int, 
                               src: str = None, dst: str = None) -> Dict:
        """Create a buffer resource definition"""
        resource = {
            "buffer": {
                "uid": uid,
                "size": size
            }
        }
        
        if src:
            resource["buffer"]["src"] = src
        if dst:
            resource["buffer"]["dst"] = dst
            
        return resource
    
    def _create_shader_resource(self, uid: str, shader_path: str) -> Dict:
        """Create a shader resource definition"""
        return {
            "shader": {
                "uid": uid,
                "src": shader_path
            }
        }
    
    def _create_compute_dispatch(self, shader_ref: str, 
                                bindings: List[Dict],
                                workgroup: List[int]) -> Dict:
        """Create a compute dispatch command"""
        return {
            "dispatch_compute": {
                "shader_ref": shader_ref,
                "bindings": bindings,
                "workgroup": workgroup
            }
        }
    
    def _add_conv2d_resources(self, scenario: Dict, 
                             input_shape: Tuple[int, ...],
                             params: Dict) -> Dict:
        """Add Conv2D specific resources"""
        batch, height, width, in_channels = input_shape
        
        # Default parameters
        kernel_size = params.get("kernel_size", 3)
        out_channels = params.get("out_channels", 32)
        stride = params.get("stride", 1)
        padding = params.get("padding", 1)
        
        # Calculate output shape
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1
        
        # Add weight tensor
        weight_resource = self._create_tensor_resource(
            uid="_weights",
            dims=[kernel_size, kernel_size, in_channels, out_channels],
            format="VK_FORMAT_R32_SFLOAT",
            access="readonly",
            src="conv2d_weights.npy"
        )
        scenario["resources"].append(weight_resource)
        
        # Add bias tensor
        bias_resource = self._create_tensor_resource(
            uid="_bias",
            dims=[out_channels],
            format="VK_FORMAT_R32_SFLOAT",
            access="readonly",
            src="conv2d_bias.npy"
        )
        scenario["resources"].append(bias_resource)
        
        # Add output tensor
        output_resource = self._create_tensor_resource(
            uid="_output",
            dims=[batch, out_height, out_width, out_channels],
            format="VK_FORMAT_R32_SFLOAT",
            access="writeonly",
            dst="conv2d_output.npy"
        )
        scenario["resources"].append(output_resource)
        
        # Add shader
        shader_resource = self._create_shader_resource(
            uid="_conv2d_shader",
            shader_path="optimized_conv2d.spv"
        )
        scenario["resources"].append(shader_resource)
        
        # Add compute dispatch
        dispatch = self._create_compute_dispatch(
            shader_ref="_conv2d_shader",
            bindings=[
                {"tensor": "_input"},
                {"tensor": "_weights"},
                {"tensor": "_bias"},
                {"tensor": "_output"}
            ],
            workgroup=[8, 8, 1]
        )
        scenario["commands"].append(dispatch)
        
        return scenario
    
    def _add_matmul_resources(self, scenario: Dict,
                            input_shape: Tuple[int, ...],
                            params: Dict) -> Dict:
        """Add MatMul specific resources"""
        m, k = input_shape  # M x K matrix
        n = params.get("n", k)  # Output dimension
        
        # Add second matrix
        matrix_b = self._create_tensor_resource(
            uid="_matrix_b",
            dims=[k, n],
            format="VK_FORMAT_R32_SFLOAT",
            access="readonly",
            src="matrix_b.npy"
        )
        scenario["resources"].append(matrix_b)
        
        # Add output matrix
        output = self._create_tensor_resource(
            uid="_output",
            dims=[m, n],
            format="VK_FORMAT_R32_SFLOAT",
            access="writeonly",
            dst="matmul_output.npy"
        )
        scenario["resources"].append(output)
        
        # Add shader
        shader = self._create_shader_resource(
            uid="_matmul_shader",
            shader_path="matrix_multiply.spv"
        )
        scenario["resources"].append(shader)
        
        # Add dispatch
        dispatch = self._create_compute_dispatch(
            shader_ref="_matmul_shader",
            bindings=[
                {"tensor": "_input"},
                {"tensor": "_matrix_b"},
                {"tensor": "_output"}
            ],
            workgroup=[16, 16, 1]
        )
        scenario["commands"].append(dispatch)
        
        return scenario
    
    def _add_activation_resources(self, scenario: Dict,
                                 op_type: OperationType,
                                 input_shape: Tuple[int, ...]) -> Dict:
        """Add activation function resources"""
        
        # Add output tensor (same shape as input)
        output = self._create_tensor_resource(
            uid="_output",
            dims=list(input_shape),
            format="VK_FORMAT_R32_SFLOAT",
            access="writeonly",
            dst=f"{op_type.value}_output.npy"
        )
        scenario["resources"].append(output)
        
        # Add shader
        shader_name = self.shader_map.get(op_type, op_type.value)
        shader = self._create_shader_resource(
            uid=f"_{op_type.value}_shader",
            shader_path=f"{shader_name}.spv"
        )
        scenario["resources"].append(shader)
        
        # Calculate workgroup size
        total_elements = np.prod(input_shape)
        workgroup_size = min(256, total_elements)
        num_workgroups = (total_elements + workgroup_size - 1) // workgroup_size
        
        # Add dispatch
        dispatch = self._create_compute_dispatch(
            shader_ref=f"_{op_type.value}_shader",
            bindings=[
                {"tensor": "_input"},
                {"tensor": "_output"}
            ],
            workgroup=[num_workgroups, 1, 1]
        )
        scenario["commands"].append(dispatch)
        
        return scenario
    
    def _add_pooling_resources(self, scenario: Dict,
                              op_type: OperationType,
                              input_shape: Tuple[int, ...],
                              params: Dict) -> Dict:
        """Add pooling operation resources"""
        batch, height, width, channels = input_shape
        
        # Default parameters
        pool_size = params.get("pool_size", 2)
        stride = params.get("stride", pool_size)
        
        # Calculate output shape
        out_height = height // stride
        out_width = width // stride
        
        # Add output tensor
        output = self._create_tensor_resource(
            uid="_output",
            dims=[batch, out_height, out_width, channels],
            format="VK_FORMAT_R32_SFLOAT",
            access="writeonly",
            dst=f"{op_type.value}_output.npy"
        )
        scenario["resources"].append(output)
        
        # Add shader
        shader_name = self.shader_map.get(op_type, op_type.value)
        shader = self._create_shader_resource(
            uid=f"_{op_type.value}_shader",
            shader_path=f"{shader_name}.spv"
        )
        scenario["resources"].append(shader)
        
        # Add dispatch
        dispatch = self._create_compute_dispatch(
            shader_ref=f"_{op_type.value}_shader",
            bindings=[
                {"tensor": "_input"},
                {"tensor": "_output"}
            ],
            workgroup=[out_width, out_height, 1]
        )
        scenario["commands"].append(dispatch)
        
        return scenario
    
    def _add_binary_op_resources(self, scenario: Dict,
                                op_type: OperationType,
                                input_shape: Tuple[int, ...]) -> Dict:
        """Add binary operation resources (add, multiply)"""
        
        # Add second input tensor
        input2 = self._create_tensor_resource(
            uid="_input2",
            dims=list(input_shape),
            format="VK_FORMAT_R32_SFLOAT",
            access="readonly",
            src="input2.npy"
        )
        scenario["resources"].append(input2)
        
        # Add output tensor
        output = self._create_tensor_resource(
            uid="_output",
            dims=list(input_shape),
            format="VK_FORMAT_R32_SFLOAT",
            access="writeonly",
            dst=f"{op_type.value}_output.npy"
        )
        scenario["resources"].append(output)
        
        # Add shader
        shader_name = self.shader_map.get(op_type, op_type.value)
        shader = self._create_shader_resource(
            uid=f"_{op_type.value}_shader",
            shader_path=f"{shader_name}.spv"
        )
        scenario["resources"].append(shader)
        
        # Calculate workgroup
        total_elements = np.prod(input_shape)
        workgroup_size = min(256, total_elements)
        num_workgroups = (total_elements + workgroup_size - 1) // workgroup_size
        
        # Add dispatch
        dispatch = self._create_compute_dispatch(
            shader_ref=f"_{op_type.value}_shader",
            bindings=[
                {"tensor": "_input"},
                {"tensor": "_input2"},
                {"tensor": "_output"}
            ],
            workgroup=[num_workgroups, 1, 1]
        )
        scenario["commands"].append(dispatch)
        
        return scenario
    
    def generate_edge_case_scenarios(self) -> List[Dict]:
        """Generate scenarios for edge cases"""
        scenarios = []
        
        # Very small tensors
        small_scenario = self.generate_operation_scenario(
            OperationType.ADD,
            input_shape=(1, 1, 1, 1),
            params={}
        )
        small_scenario["name"] = "edge_case_tiny_tensor"
        scenarios.append(small_scenario)
        
        # Large tensors
        large_scenario = self.generate_operation_scenario(
            OperationType.MATMUL,
            input_shape=(2048, 2048),
            params={"n": 2048}
        )
        large_scenario["name"] = "edge_case_large_matmul"
        scenarios.append(large_scenario)
        
        # Non-power-of-2 dimensions
        odd_scenario = self.generate_operation_scenario(
            OperationType.CONV2D,
            input_shape=(1, 227, 227, 3),
            params={"kernel_size": 7, "out_channels": 96}
        )
        odd_scenario["name"] = "edge_case_odd_dimensions"
        scenarios.append(odd_scenario)
        
        # Single channel
        single_channel = self.generate_operation_scenario(
            OperationType.RELU,
            input_shape=(1, 224, 224, 1),
            params={}
        )
        single_channel["name"] = "edge_case_single_channel"
        scenarios.append(single_channel)
        
        return scenarios
    
    def generate_benchmark_scenarios(self) -> List[Dict]:
        """Generate scenarios for performance benchmarking"""
        scenarios = []
        
        # Conv2D benchmarks - different sizes
        conv_sizes = [
            (1, 224, 224, 3),   # ImageNet size
            (1, 112, 112, 64),  # Intermediate layer
            (1, 56, 56, 128),   # Deeper layer
            (1, 28, 28, 256),   # Even deeper
        ]
        
        for i, shape in enumerate(conv_sizes):
            scenario = self.generate_operation_scenario(
                OperationType.CONV2D,
                input_shape=shape,
                params={"kernel_size": 3, "out_channels": shape[3] * 2}
            )
            scenario["name"] = f"benchmark_conv2d_size_{i}"
            scenarios.append(scenario)
        
        # MatMul benchmarks - different sizes
        matmul_sizes = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ]
        
        for size in matmul_sizes:
            scenario = self.generate_operation_scenario(
                OperationType.MATMUL,
                input_shape=size,
                params={"n": size[1]}
            )
            scenario["name"] = f"benchmark_matmul_{size[0]}x{size[1]}"
            scenarios.append(scenario)
        
        # Activation benchmarks
        activation_shape = (1, 224, 224, 64)
        for op in [OperationType.RELU, OperationType.SIGMOID, OperationType.TANH]:
            scenario = self.generate_operation_scenario(
                op,
                input_shape=activation_shape,
                params={}
            )
            scenario["name"] = f"benchmark_{op.value}"
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_model_scenarios(self) -> List[Dict]:
        """Generate scenarios for complete models"""
        scenarios = []
        
        # MobileNet-like scenario
        mobilenet_scenario = {
            "name": "model_mobilenet_inference",
            "description": "MobileNet-style inference pipeline",
            "resources": [],
            "commands": []
        }
        
        # Add input image
        input_tensor = self._create_tensor_resource(
            uid="_input_image",
            dims=[1, 224, 224, 3],
            format="VK_FORMAT_R32_SFLOAT",
            access="readonly",
            src="input_image.npy"
        )
        mobilenet_scenario["resources"].append(input_tensor)
        
        # Add multiple conv layers (simplified)
        current_input = "_input_image"
        layer_configs = [
            {"in_shape": [1, 224, 224, 3], "out_channels": 32, "stride": 2},
            {"in_shape": [1, 112, 112, 32], "out_channels": 64, "stride": 1},
            {"in_shape": [1, 112, 112, 64], "out_channels": 128, "stride": 2},
        ]
        
        for i, config in enumerate(layer_configs):
            # Create intermediate tensors and operations
            output_uid = f"_layer_{i}_output"
            in_shape = config["in_shape"]
            out_h = in_shape[1] // config["stride"]
            out_w = in_shape[2] // config["stride"]
            
            output_tensor = self._create_tensor_resource(
                uid=output_uid,
                dims=[1, out_h, out_w, config["out_channels"]],
                format="VK_FORMAT_R32_SFLOAT",
                access="writeonly" if i < len(layer_configs) - 1 else "writeonly",
                dst=f"layer_{i}_output.npy" if i == len(layer_configs) - 1 else None
            )
            mobilenet_scenario["resources"].append(output_tensor)
            
            current_input = output_uid
        
        scenarios.append(mobilenet_scenario)
        
        return scenarios
    
    def save_scenario(self, scenario: Dict, filename: str = None) -> Path:
        """Save scenario to JSON file"""
        if filename is None:
            filename = f"{scenario['name']}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(scenario, f, indent=2)
        
        return filepath
    
    def generate_all_scenarios(self) -> Dict[str, List[Path]]:
        """Generate all test scenarios"""
        all_scenarios = {
            "operations": [],
            "edge_cases": [],
            "benchmarks": [],
            "models": []
        }
        
        # Generate operation scenarios
        operations = [
            (OperationType.CONV2D, (1, 56, 56, 64), {"kernel_size": 3, "out_channels": 128}),
            (OperationType.MATMUL, (512, 512), {"n": 512}),
            (OperationType.RELU, (1, 224, 224, 64), {}),
            (OperationType.MAXPOOL, (1, 112, 112, 64), {"pool_size": 2}),
            (OperationType.ADD, (1, 56, 56, 128), {}),
        ]
        
        for op_type, shape, params in operations:
            scenario = self.generate_operation_scenario(op_type, shape, params)
            path = self.save_scenario(scenario)
            all_scenarios["operations"].append(path)
        
        # Generate edge cases
        for scenario in self.generate_edge_case_scenarios():
            path = self.save_scenario(scenario)
            all_scenarios["edge_cases"].append(path)
        
        # Generate benchmarks
        for scenario in self.generate_benchmark_scenarios():
            path = self.save_scenario(scenario)
            all_scenarios["benchmarks"].append(path)
        
        # Generate model scenarios
        for scenario in self.generate_model_scenarios():
            path = self.save_scenario(scenario)
            all_scenarios["models"].append(path)
        
        print(f"Generated {sum(len(v) for v in all_scenarios.values())} scenarios")
        print(f"Saved to: {self.output_dir}")
        
        return all_scenarios

def main():
    """Generate test scenarios"""
    generator = ScenarioGenerator()
    scenarios = generator.generate_all_scenarios()
    
    print("\nGenerated Scenarios:")
    for category, paths in scenarios.items():
        print(f"\n{category.title()}:")
        for path in paths:
            print(f"  - {path.name}")

if __name__ == "__main__":
    main()