#!/usr/bin/env python3
"""
Test Scenario Generator for Vulkan ML SDK
Generates comprehensive test scenarios for all ML operations
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import random

class OperationType(Enum):
    CONV2D = "conv2d"
    MATMUL = "matmul"
    MAXPOOL = "maxpool"
    AVGPOOL = "avgpool"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    BATCHNORM = "batchnorm"
    LAYERNORM = "layernorm"
    ADD = "add"
    MULTIPLY = "multiply"
    CONCAT = "concat"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"

class DataType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT32 = "int32"

class ScenarioGenerator:
    """Generate test scenarios for ML operations"""
    
    def __init__(self, output_dir: str = "/Users/jerry/Vulkan/tests/data/scenarios"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Operation configurations
        self.op_configs = {
            OperationType.CONV2D: self._conv2d_configs,
            OperationType.MATMUL: self._matmul_configs,
            OperationType.MAXPOOL: self._pooling_configs,
            OperationType.AVGPOOL: self._pooling_configs,
            OperationType.RELU: self._activation_configs,
            OperationType.SIGMOID: self._activation_configs,
            OperationType.TANH: self._activation_configs,
            OperationType.SOFTMAX: self._softmax_configs,
            OperationType.BATCHNORM: self._normalization_configs,
            OperationType.LAYERNORM: self._normalization_configs,
            OperationType.ADD: self._elementwise_configs,
            OperationType.MULTIPLY: self._elementwise_configs,
            OperationType.CONCAT: self._concat_configs,
            OperationType.RESHAPE: self._reshape_configs,
            OperationType.TRANSPOSE: self._transpose_configs,
            OperationType.QUANTIZE: self._quantization_configs,
            OperationType.DEQUANTIZE: self._quantization_configs
        }
    
    def generate_all_scenarios(self) -> List[str]:
        """Generate scenarios for all operation types"""
        scenarios = []
        
        for op_type in OperationType:
            op_scenarios = self.generate_operation_scenarios(op_type)
            scenarios.extend(op_scenarios)
        
        # Generate composite scenarios
        scenarios.extend(self.generate_composite_scenarios())
        
        # Generate edge case scenarios
        scenarios.extend(self.generate_edge_case_scenarios())
        
        # Generate stress test scenarios
        scenarios.extend(self.generate_stress_scenarios())
        
        return scenarios
    
    def generate_operation_scenarios(self, op_type: OperationType) -> List[str]:
        """Generate scenarios for a specific operation"""
        scenarios = []
        config_fn = self.op_configs.get(op_type)
        
        if not config_fn:
            return scenarios
        
        configs = config_fn()
        
        for i, config in enumerate(configs):
            scenario = self._create_scenario(op_type, config, f"{op_type.value}_{i}")
            filename = f"{op_type.value}_test_{i}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(scenario, f, indent=2)
            
            scenarios.append(str(filepath))
        
        return scenarios
    
    def _create_scenario(self, op_type: OperationType, config: Dict[str, Any], 
                        name: str) -> Dict[str, Any]:
        """Create a scenario JSON structure"""
        return {
            "name": f"Test_{name}",
            "description": f"Test scenario for {op_type.value} operation",
            "version": "1.0",
            "operations": [
                {
                    "type": op_type.value,
                    "config": config,
                    "input_shapes": config.get("input_shapes", [[1, 224, 224, 3]]),
                    "output_validation": {
                        "enabled": True,
                        "tolerance": 1e-5,
                        "reference_impl": "numpy"
                    }
                }
            ],
            "performance_targets": {
                "latency_ms": config.get("target_latency", 10),
                "throughput_ops": config.get("target_throughput", 1000),
                "memory_mb": config.get("target_memory", 100)
            }
        }
    
    def _conv2d_configs(self) -> List[Dict[str, Any]]:
        """Generate Conv2D test configurations"""
        configs = []
        
        # Standard configurations
        kernel_sizes = [(1, 1), (3, 3), (5, 5), (7, 7)]
        strides = [(1, 1), (2, 2)]
        paddings = ["same", "valid"]
        channels = [32, 64, 128, 256]
        
        for kernel in kernel_sizes:
            for stride in strides:
                for padding in paddings:
                    for in_channels, out_channels in zip(channels[:-1], channels[1:]):
                        configs.append({
                            "kernel_size": kernel,
                            "stride": stride,
                            "padding": padding,
                            "in_channels": in_channels,
                            "out_channels": out_channels,
                            "input_shapes": [[1, 224, 224, in_channels]],
                            "activation": "relu",
                            "use_bias": True
                        })
        
        # Depthwise configurations
        configs.append({
            "kernel_size": [3, 3],
            "stride": [1, 1],
            "padding": "same",
            "groups": 32,
            "in_channels": 32,
            "out_channels": 32,
            "input_shapes": [[1, 112, 112, 32]],
            "type": "depthwise"
        })
        
        # Dilated convolutions
        for dilation in [2, 4]:
            configs.append({
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": "same",
                "dilation": [dilation, dilation],
                "in_channels": 64,
                "out_channels": 64,
                "input_shapes": [[1, 56, 56, 64]]
            })
        
        return configs
    
    def _matmul_configs(self) -> List[Dict[str, Any]]:
        """Generate MatMul test configurations"""
        configs = []
        
        # Square matrices
        for size in [16, 32, 64, 128, 256, 512, 1024]:
            configs.append({
                "m": size,
                "n": size,
                "k": size,
                "input_shapes": [[1, size, size], [1, size, size]],
                "transpose_a": False,
                "transpose_b": False
            })
        
        # Rectangular matrices
        shapes = [
            (128, 256, 64),
            (256, 128, 512),
            (1024, 768, 256),
            (768, 1024, 512)
        ]
        
        for m, n, k in shapes:
            configs.append({
                "m": m,
                "n": n,
                "k": k,
                "input_shapes": [[1, m, k], [1, k, n]],
                "transpose_a": False,
                "transpose_b": False
            })
        
        # Batch matmul
        for batch in [2, 4, 8]:
            configs.append({
                "batch": batch,
                "m": 128,
                "n": 128,
                "k": 128,
                "input_shapes": [[batch, 128, 128], [batch, 128, 128]]
            })
        
        return configs
    
    def _pooling_configs(self) -> List[Dict[str, Any]]:
        """Generate pooling test configurations"""
        configs = []
        
        pool_sizes = [(2, 2), (3, 3), (5, 5)]
        strides = [(1, 1), (2, 2)]
        paddings = ["valid", "same"]
        
        for pool_size in pool_sizes:
            for stride in strides:
                for padding in paddings:
                    configs.append({
                        "pool_size": pool_size,
                        "stride": stride,
                        "padding": padding,
                        "input_shapes": [[1, 224, 224, 64]]
                    })
        
        # Global pooling
        configs.append({
            "global_pooling": True,
            "input_shapes": [[1, 7, 7, 512]]
        })
        
        return configs
    
    def _activation_configs(self) -> List[Dict[str, Any]]:
        """Generate activation function test configurations"""
        configs = []
        
        input_shapes = [
            [1, 1000],
            [1, 224, 224, 3],
            [8, 512],
            [1, 7, 7, 512]
        ]
        
        for shape in input_shapes:
            configs.append({
                "input_shapes": [shape],
                "negative_slope": 0.01  # For LeakyReLU variant
            })
        
        return configs
    
    def _softmax_configs(self) -> List[Dict[str, Any]]:
        """Generate softmax test configurations"""
        configs = []
        
        # Different axes
        for axis in [-1, 0, 1]:
            configs.append({
                "axis": axis,
                "input_shapes": [[1, 1000]],
                "temperature": 1.0
            })
        
        # Different shapes
        shapes = [[1, 10], [8, 1000], [1, 21, 21, 256]]
        for shape in shapes:
            configs.append({
                "axis": -1,
                "input_shapes": [shape]
            })
        
        return configs
    
    def _normalization_configs(self) -> List[Dict[str, Any]]:
        """Generate normalization test configurations"""
        configs = []
        
        # BatchNorm configs
        for channels in [32, 64, 128, 256]:
            configs.append({
                "num_features": channels,
                "eps": 1e-5,
                "momentum": 0.1,
                "input_shapes": [[8, 56, 56, channels]],
                "training": False
            })
        
        # LayerNorm configs
        for dim in [256, 512, 768, 1024]:
            configs.append({
                "normalized_shape": [dim],
                "eps": 1e-6,
                "input_shapes": [[1, 128, dim]]
            })
        
        return configs
    
    def _elementwise_configs(self) -> List[Dict[str, Any]]:
        """Generate elementwise operation configs"""
        configs = []
        
        shapes = [
            [[1, 1000], [1, 1000]],
            [[8, 224, 224, 3], [8, 224, 224, 3]],
            [[1, 512], [1, 512]],
            [[1, 1], [1, 1000]]  # Broadcasting
        ]
        
        for shape_pair in shapes:
            configs.append({
                "input_shapes": shape_pair,
                "alpha": 1.0,
                "beta": 1.0
            })
        
        return configs
    
    def _concat_configs(self) -> List[Dict[str, Any]]:
        """Generate concatenation configs"""
        configs = []
        
        # Different axes
        for axis in [0, 1, 2, 3]:
            configs.append({
                "axis": axis,
                "input_shapes": [
                    [1, 56, 56, 64],
                    [1, 56, 56, 128],
                    [1, 56, 56, 64]
                ]
            })
        
        return configs
    
    def _reshape_configs(self) -> List[Dict[str, Any]]:
        """Generate reshape configs"""
        configs = []
        
        reshape_pairs = [
            ([1, 784], [1, 28, 28, 1]),
            ([8, 224, 224, 3], [8, -1]),
            ([1, 512], [1, 1, 1, 512]),
            ([100, 10], [10, 10, 10])
        ]
        
        for input_shape, output_shape in reshape_pairs:
            configs.append({
                "input_shapes": [input_shape],
                "output_shape": output_shape
            })
        
        return configs
    
    def _transpose_configs(self) -> List[Dict[str, Any]]:
        """Generate transpose configs"""
        configs = []
        
        perm_configs = [
            ([0, 2, 3, 1], [1, 224, 224, 3]),  # NHWC to NCHW
            ([0, 3, 1, 2], [1, 3, 224, 224]),  # NCHW to NHWC
            ([1, 0], [100, 256]),
            ([0, 2, 1], [8, 128, 512])
        ]
        
        for perm, shape in perm_configs:
            configs.append({
                "perm": perm,
                "input_shapes": [shape]
            })
        
        return configs
    
    def _quantization_configs(self) -> List[Dict[str, Any]]:
        """Generate quantization/dequantization configs"""
        configs = []
        
        # Different quantization schemes
        for dtype in ["int8", "uint8"]:
            for symmetric in [True, False]:
                configs.append({
                    "dtype": dtype,
                    "symmetric": symmetric,
                    "scale": 0.1,
                    "zero_point": 128 if not symmetric else 0,
                    "input_shapes": [[1, 224, 224, 3]]
                })
        
        return configs
    
    def generate_composite_scenarios(self) -> List[str]:
        """Generate scenarios with multiple operations"""
        scenarios = []
        
        # MobileNet-like block
        mobilenet_scenario = {
            "name": "MobileNet_Block",
            "description": "MobileNet-style inverted residual block",
            "operations": [
                {"type": "conv2d", "config": {"kernel_size": [1, 1], "stride": [1, 1]}},
                {"type": "batchnorm"},
                {"type": "relu"},
                {"type": "conv2d", "config": {"kernel_size": [3, 3], "stride": [2, 2], "groups": 32}},
                {"type": "batchnorm"},
                {"type": "relu"},
                {"type": "conv2d", "config": {"kernel_size": [1, 1], "stride": [1, 1]}},
                {"type": "batchnorm"}
            ]
        }
        
        # ResNet-like block
        resnet_scenario = {
            "name": "ResNet_Block",
            "description": "ResNet-style residual block",
            "operations": [
                {"type": "conv2d", "config": {"kernel_size": [3, 3], "stride": [1, 1]}},
                {"type": "batchnorm"},
                {"type": "relu"},
                {"type": "conv2d", "config": {"kernel_size": [3, 3], "stride": [1, 1]}},
                {"type": "batchnorm"},
                {"type": "add"},  # Residual connection
                {"type": "relu"}
            ]
        }
        
        # Transformer-like attention
        attention_scenario = {
            "name": "Attention_Block",
            "description": "Simplified attention mechanism",
            "operations": [
                {"type": "matmul"},  # Q @ K^T
                {"type": "multiply", "config": {"alpha": 0.125}},  # Scale
                {"type": "softmax", "config": {"axis": -1}},
                {"type": "matmul"}   # @ V
            ]
        }
        
        for scenario in [mobilenet_scenario, resnet_scenario, attention_scenario]:
            filename = f"composite_{scenario['name'].lower()}.json"
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(scenario, f, indent=2)
            scenarios.append(str(filepath))
        
        return scenarios
    
    def generate_edge_case_scenarios(self) -> List[str]:
        """Generate edge case test scenarios"""
        scenarios = []
        
        edge_cases = [
            {
                "name": "Zero_Input",
                "description": "Test with zero inputs",
                "operations": [
                    {"type": "conv2d", "input_data": "zeros", "input_shapes": [[1, 224, 224, 3]]}
                ]
            },
            {
                "name": "Large_Values",
                "description": "Test with very large values",
                "operations": [
                    {"type": "relu", "input_data": "large", "input_shapes": [[1, 1000]],
                     "input_range": [1e6, 1e7]}
                ]
            },
            {
                "name": "Small_Values",
                "description": "Test with very small values",
                "operations": [
                    {"type": "sigmoid", "input_data": "small", "input_shapes": [[1, 1000]],
                     "input_range": [-1e-7, 1e-7]}
                ]
            },
            {
                "name": "NaN_Handling",
                "description": "Test NaN propagation",
                "operations": [
                    {"type": "add", "input_data": "contains_nan", "input_shapes": [[1, 100]]}
                ]
            },
            {
                "name": "Inf_Handling",
                "description": "Test infinity handling",
                "operations": [
                    {"type": "multiply", "input_data": "contains_inf", "input_shapes": [[1, 100]]}
                ]
            },
            {
                "name": "Single_Element",
                "description": "Test with single element tensors",
                "operations": [
                    {"type": "relu", "input_shapes": [[1, 1]]}
                ]
            },
            {
                "name": "Empty_Batch",
                "description": "Test with zero batch size",
                "operations": [
                    {"type": "conv2d", "input_shapes": [[0, 224, 224, 3]]}
                ]
            }
        ]
        
        for edge_case in edge_cases:
            filename = f"edge_{edge_case['name'].lower()}.json"
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(edge_case, f, indent=2)
            scenarios.append(str(filepath))
        
        return scenarios
    
    def generate_stress_scenarios(self) -> List[str]:
        """Generate stress test scenarios"""
        scenarios = []
        
        stress_tests = [
            {
                "name": "Large_Conv2D",
                "description": "Very large convolution",
                "operations": [
                    {"type": "conv2d", 
                     "config": {"kernel_size": [11, 11], "in_channels": 512, "out_channels": 512},
                     "input_shapes": [[1, 512, 512, 512]]}
                ]
            },
            {
                "name": "Large_MatMul",
                "description": "Very large matrix multiplication",
                "operations": [
                    {"type": "matmul",
                     "config": {"m": 4096, "n": 4096, "k": 4096},
                     "input_shapes": [[1, 4096, 4096], [1, 4096, 4096]]}
                ]
            },
            {
                "name": "Deep_Network",
                "description": "Very deep network simulation",
                "operations": [
                    {"type": "conv2d", "config": {"kernel_size": [3, 3]}}
                    for _ in range(100)  # 100 conv layers
                ]
            },
            {
                "name": "High_Batch",
                "description": "High batch size test",
                "operations": [
                    {"type": "conv2d", "input_shapes": [[128, 224, 224, 3]]}
                ]
            },
            {
                "name": "Memory_Intensive",
                "description": "Memory intensive operations",
                "operations": [
                    {"type": "concat", 
                     "input_shapes": [[1, 1024, 1024, 256] for _ in range(10)],
                     "axis": 3}
                ]
            }
        ]
        
        for stress_test in stress_tests:
            filename = f"stress_{stress_test['name'].lower()}.json"
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(stress_test, f, indent=2)
            scenarios.append(str(filepath))
        
        return scenarios
    
    def generate_random_scenario(self, num_ops: int = 10) -> Dict[str, Any]:
        """Generate a random test scenario"""
        operations = []
        
        for i in range(num_ops):
            op_type = random.choice(list(OperationType))
            config_fn = self.op_configs.get(op_type)
            
            if config_fn:
                configs = config_fn()
                if configs:
                    config = random.choice(configs)
                    operations.append({
                        "type": op_type.value,
                        "config": config
                    })
        
        return {
            "name": f"Random_Scenario_{random.randint(1000, 9999)}",
            "description": "Randomly generated test scenario",
            "operations": operations
        }
    
    def validate_scenario(self, scenario_path: str) -> bool:
        """Validate a scenario file"""
        try:
            with open(scenario_path, 'r') as f:
                scenario = json.load(f)
            
            # Check required fields
            required = ["name", "operations"]
            for field in required:
                if field not in scenario:
                    print(f"Missing required field: {field}")
                    return False
            
            # Validate operations
            for op in scenario["operations"]:
                if "type" not in op:
                    print(f"Operation missing type field")
                    return False
                
                # Check if operation type is valid
                try:
                    OperationType(op["type"])
                except ValueError:
                    print(f"Invalid operation type: {op['type']}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False

if __name__ == "__main__":
    generator = ScenarioGenerator()
    
    # Generate all scenarios
    print("Generating test scenarios...")
    scenarios = generator.generate_all_scenarios()
    
    print(f"Generated {len(scenarios)} test scenarios")
    print(f"Scenarios saved to: {generator.output_dir}")
    
    # Validate generated scenarios
    valid_count = sum(1 for s in scenarios if generator.validate_scenario(s))
    print(f"Validated: {valid_count}/{len(scenarios)} scenarios")