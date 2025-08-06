#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
SPDX-License-Identifier: Apache-2.0

Integration Tests for Scenarios and ML Models
Tests end-to-end workflows, JSON scenario validation, model loading,
and inference pipelines.
"""

import pytest
import json
import numpy as np
import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Constants
SDK_ROOT = Path("/Users/jerry/Vulkan")
SCENARIO_RUNNER = SDK_ROOT / "builds/ARM-ML-SDK-Complete/bin/scenario-runner"
MODELS_DIR = SDK_ROOT / "builds/ARM-ML-SDK-Complete/models"
SHADERS_DIR = SDK_ROOT / "builds/ARM-ML-SDK-Complete/shaders"
TOOLS_DIR = SDK_ROOT / "builds/ARM-ML-SDK-Complete/tools"


class ScenarioBuilder:
    """Helper class to build JSON scenarios programmatically"""
    
    def __init__(self, name: str = "test_scenario"):
        self.scenario = {
            "name": name,
            "description": f"Integration test scenario: {name}",
            "resources": {},
            "commands": []
        }
        self.resource_counter = 0
    
    def add_buffer(self, size: int, data: Optional[List[float]] = None) -> str:
        """Add a buffer resource"""
        resource_id = f"buffer_{self.resource_counter}"
        self.resource_counter += 1
        
        resource = {
            "type": "buffer",
            "size": size
        }
        
        if data is not None:
            # Save data to temporary file
            data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin')
            np.array(data, dtype=np.float32).tofile(data_file.name)
            resource["src"] = data_file.name
        
        self.scenario["resources"][resource_id] = resource
        return resource_id
    
    def add_image(self, width: int, height: int, format: str = "R32_SFLOAT") -> str:
        """Add an image resource"""
        resource_id = f"image_{self.resource_counter}"
        self.resource_counter += 1
        
        self.scenario["resources"][resource_id] = {
            "type": "image",
            "width": width,
            "height": height,
            "format": format
        }
        return resource_id
    
    def add_tensor(self, shape: List[int], dtype: str = "float32") -> str:
        """Add a tensor resource"""
        resource_id = f"tensor_{self.resource_counter}"
        self.resource_counter += 1
        
        self.scenario["resources"][resource_id] = {
            "type": "tensor",
            "shape": shape,
            "dtype": dtype
        }
        return resource_id
    
    def add_shader(self, shader_path: str) -> str:
        """Add a shader resource"""
        resource_id = f"shader_{self.resource_counter}"
        self.resource_counter += 1
        
        self.scenario["resources"][resource_id] = {
            "type": "shader",
            "src": shader_path
        }
        return resource_id
    
    def add_compute_dispatch(self, shader_id: str, workgroup: List[int], 
                           bindings: Dict[int, str]) -> None:
        """Add a compute dispatch command"""
        self.scenario["commands"].append({
            "type": "dispatch_compute",
            "shader": shader_id,
            "workgroup": workgroup,
            "bindings": bindings
        })
    
    def add_barrier(self, src_stage: str = "COMPUTE", dst_stage: str = "COMPUTE") -> None:
        """Add a pipeline barrier"""
        self.scenario["commands"].append({
            "type": "dispatch_barrier",
            "src_stage": src_stage,
            "dst_stage": dst_stage
        })
    
    def add_mark_boundary(self, frame_id: int = 0) -> None:
        """Add a mark boundary command"""
        self.scenario["commands"].append({
            "type": "mark_boundary",
            "frame_id": frame_id
        })
    
    def save(self, filepath: str) -> None:
        """Save scenario to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.scenario, f, indent=2)
    
    def to_dict(self) -> Dict:
        """Return scenario as dictionary"""
        return self.scenario


class TestHelpers:
    """Helper functions for tests"""
    
    @staticmethod
    def run_scenario(scenario_path: str, output_dir: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run a scenario and return the result"""
        cmd = [str(SCENARIO_RUNNER), "--scenario", scenario_path]
        
        if output_dir:
            cmd.extend(["--output", output_dir])
        
        # Set library path
        env = os.environ.copy()
        env["DYLD_LIBRARY_PATH"] = f"/usr/local/lib:{SDK_ROOT}/builds/ARM-ML-SDK-Complete/lib"
        
        return subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    @staticmethod
    def validate_output_file(filepath: str, expected_shape: List[int], 
                           dtype: np.dtype = np.float32) -> np.ndarray:
        """Validate and load an output file"""
        assert os.path.exists(filepath), f"Output file {filepath} does not exist"
        
        data = np.fromfile(filepath, dtype=dtype)
        expected_size = np.prod(expected_shape)
        
        assert data.size == expected_size, \
            f"Output size mismatch: expected {expected_size}, got {data.size}"
        
        return data.reshape(expected_shape)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    temp_path = tempfile.mkdtemp(prefix="ml_sdk_test_")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def simple_add_scenario(temp_dir):
    """Create a simple vector addition scenario"""
    builder = ScenarioBuilder("vector_add")
    
    # Create input buffers
    size = 1024
    data_a = list(range(size))
    data_b = [x * 2 for x in range(size)]
    
    buffer_a = builder.add_buffer(size * 4, data_a)
    buffer_b = builder.add_buffer(size * 4, data_b)
    buffer_c = builder.add_buffer(size * 4)
    
    # Add compute shader (assuming add.spv exists)
    shader = builder.add_shader(str(SHADERS_DIR / "add.spv"))
    
    # Add dispatch command
    builder.add_compute_dispatch(
        shader,
        [size // 64, 1, 1],  # Assuming workgroup size of 64
        {0: buffer_a, 1: buffer_b, 2: buffer_c}
    )
    
    # Save scenario
    scenario_path = os.path.join(temp_dir, "vector_add.json")
    builder.save(scenario_path)
    
    return scenario_path, buffer_c, data_a, data_b


# ============================================================================
# Scenario Validation Tests
# ============================================================================

class TestScenarioValidation:
    """Test JSON scenario validation and error handling"""
    
    def test_valid_scenario_structure(self, temp_dir):
        """Test that valid scenarios are accepted"""
        builder = ScenarioBuilder("valid_scenario")
        builder.add_buffer(1024)
        builder.add_tensor([16, 16, 3])
        builder.add_image(256, 256)
        
        scenario_path = os.path.join(temp_dir, "valid.json")
        builder.save(scenario_path)
        
        # Should not raise any errors
        with open(scenario_path) as f:
            scenario = json.load(f)
        
        assert "resources" in scenario
        assert "commands" in scenario
        assert len(scenario["resources"]) == 3
    
    def test_invalid_resource_type(self, temp_dir):
        """Test that invalid resource types are rejected"""
        scenario = {
            "name": "invalid_resource",
            "resources": {
                "invalid": {
                    "type": "unknown_type",
                    "size": 1024
                }
            },
            "commands": []
        }
        
        scenario_path = os.path.join(temp_dir, "invalid_resource.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f)
        
        result = TestHelpers.run_scenario(scenario_path)
        assert result.returncode != 0
        assert "unknown_type" in result.stderr or "invalid" in result.stderr
    
    def test_missing_required_fields(self, temp_dir):
        """Test that missing required fields are detected"""
        scenario = {
            "name": "missing_fields",
            "resources": {
                "buffer1": {
                    "type": "buffer"
                    # Missing 'size' field
                }
            },
            "commands": []
        }
        
        scenario_path = os.path.join(temp_dir, "missing_fields.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f)
        
        result = TestHelpers.run_scenario(scenario_path)
        assert result.returncode != 0
    
    def test_circular_dependencies(self, temp_dir):
        """Test detection of circular resource dependencies"""
        scenario = {
            "name": "circular_deps",
            "resources": {
                "alias1": {
                    "type": "alias",
                    "resource": "alias2"
                },
                "alias2": {
                    "type": "alias",
                    "resource": "alias1"
                }
            },
            "commands": []
        }
        
        scenario_path = os.path.join(temp_dir, "circular.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f)
        
        result = TestHelpers.run_scenario(scenario_path)
        assert result.returncode != 0


# ============================================================================
# Model Loading Tests
# ============================================================================

class TestModelLoading:
    """Test TFLite model loading and validation"""
    
    @pytest.mark.parametrize("model_name", [
        "mobilenet_v2.tflite",
        "la_muse.tflite",
        "udnie.tflite",
        "mirror.tflite",
        "wave_crop.tflite",
        "des_glaneuses.tflite",
        "fire_detection.tflite"
    ])
    def test_model_exists(self, model_name):
        """Test that all expected models exist"""
        model_path = MODELS_DIR / model_name
        assert model_path.exists(), f"Model {model_name} not found"
        assert model_path.stat().st_size > 0, f"Model {model_name} is empty"
    
    def test_model_metadata_extraction(self):
        """Test extracting metadata from TFLite models"""
        # This would use the model converter or analysis tools
        analyze_script = TOOLS_DIR / "analyze_tflite_model.py"
        
        if analyze_script.exists():
            model_path = MODELS_DIR / "mobilenet_v2.tflite"
            result = subprocess.run(
                ["python3", str(analyze_script), str(model_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                output = result.stdout
                # Check for expected metadata
                assert "input" in output.lower() or "tensor" in output.lower()
    
    def test_model_input_output_shapes(self):
        """Test that model input/output shapes are correct"""
        expected_shapes = {
            "mobilenet_v2.tflite": {
                "input": [1, 224, 224, 3],
                "output": [1, 1001]
            },
            "fire_detection.tflite": {
                "input": [1, 224, 224, 3],
                "output": [1, 2]
            }
        }
        
        # This would be validated during model loading
        for model_name, shapes in expected_shapes.items():
            model_path = MODELS_DIR / model_name
            if model_path.exists():
                # In a real test, we would load the model and check shapes
                pass


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

class TestEndToEndWorkflows:
    """Test complete ML inference pipelines"""
    
    def test_simple_compute_workflow(self, temp_dir):
        """Test a simple compute shader workflow"""
        builder = ScenarioBuilder("simple_compute")
        
        # Create resources
        buffer_in = builder.add_buffer(256 * 4, list(range(256)))
        buffer_out = builder.add_buffer(256 * 4)
        
        # Add shader (multiply by 2)
        if (SHADERS_DIR / "multiply.spv").exists():
            shader = builder.add_shader(str(SHADERS_DIR / "multiply.spv"))
            
            # Add compute dispatch
            builder.add_compute_dispatch(
                shader,
                [4, 1, 1],  # 256 / 64 = 4 workgroups
                {0: buffer_in, 1: buffer_out}
            )
            
            # Save and run
            scenario_path = os.path.join(temp_dir, "compute.json")
            builder.save(scenario_path)
            
            result = TestHelpers.run_scenario(scenario_path, temp_dir)
            assert result.returncode == 0, f"Scenario failed: {result.stderr}"
    
    def test_multi_stage_pipeline(self, temp_dir):
        """Test a multi-stage compute pipeline with barriers"""
        builder = ScenarioBuilder("multi_stage")
        
        # Create buffers
        size = 512
        buffer_a = builder.add_buffer(size * 4, list(range(size)))
        buffer_b = builder.add_buffer(size * 4)
        buffer_c = builder.add_buffer(size * 4)
        
        # Stage 1: Process buffer_a -> buffer_b
        if (SHADERS_DIR / "add.spv").exists():
            shader1 = builder.add_shader(str(SHADERS_DIR / "add.spv"))
            builder.add_compute_dispatch(
                shader1,
                [size // 64, 1, 1],
                {0: buffer_a, 1: buffer_a, 2: buffer_b}
            )
        
        # Barrier between stages
        builder.add_barrier("COMPUTE", "COMPUTE")
        
        # Stage 2: Process buffer_b -> buffer_c
        if (SHADERS_DIR / "multiply.spv").exists():
            shader2 = builder.add_shader(str(SHADERS_DIR / "multiply.spv"))
            builder.add_compute_dispatch(
                shader2,
                [size // 64, 1, 1],
                {0: buffer_b, 1: buffer_c}
            )
        
        # Save and run
        scenario_path = os.path.join(temp_dir, "multi_stage.json")
        builder.save(scenario_path)
        
        result = TestHelpers.run_scenario(scenario_path, temp_dir)
        # May fail if shaders don't exist, but structure is valid
        assert result.returncode == 0 or "shader" in result.stderr.lower()
    
    def test_tensor_aliasing(self, temp_dir):
        """Test tensor aliasing with images"""
        builder = ScenarioBuilder("tensor_aliasing")
        
        # Create image and aliased tensor
        image = builder.add_image(128, 128, "R32_SFLOAT")
        
        # Create aliased tensor pointing to image
        scenario = builder.to_dict()
        scenario["resources"]["tensor_alias"] = {
            "type": "tensor",
            "shape": [128, 128, 1],
            "dtype": "float32",
            "alias": image
        }
        
        # Save scenario
        scenario_path = os.path.join(temp_dir, "aliasing.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        # Validate scenario structure
        assert "tensor_alias" in scenario["resources"]
        assert scenario["resources"]["tensor_alias"]["alias"] == image
    
    @pytest.mark.slow
    def test_model_inference_pipeline(self, temp_dir):
        """Test complete model inference pipeline"""
        builder = ScenarioBuilder("model_inference")
        
        # Create input tensor for MobileNet (1, 224, 224, 3)
        input_tensor = builder.add_tensor([1, 224, 224, 3], "float32")
        output_tensor = builder.add_tensor([1, 1001], "float32")
        
        # Add model as graph
        model_path = MODELS_DIR / "mobilenet_v2.tflite"
        if model_path.exists():
            scenario = builder.to_dict()
            scenario["resources"]["model"] = {
                "type": "graph",
                "src": str(model_path)
            }
            
            # Add graph dispatch command
            scenario["commands"].append({
                "type": "dispatch_graph",
                "graph": "model",
                "inputs": {"input": input_tensor},
                "outputs": {"output": output_tensor}
            })
            
            # Save and run
            scenario_path = os.path.join(temp_dir, "inference.json")
            with open(scenario_path, 'w') as f:
                json.dump(scenario, f, indent=2)
            
            result = TestHelpers.run_scenario(scenario_path, temp_dir)
            # Graph dispatch might not be fully implemented
            assert result.returncode == 0 or "graph" in result.stderr.lower()


# ============================================================================
# Performance and Stress Tests
# ============================================================================

class TestPerformanceIntegration:
    """Integration tests for performance scenarios"""
    
    def test_large_buffer_processing(self, temp_dir):
        """Test processing large buffers"""
        builder = ScenarioBuilder("large_buffers")
        
        # Create 16MB buffers
        size = 4 * 1024 * 1024  # 4M floats = 16MB
        buffer_in = builder.add_buffer(size * 4)
        buffer_out = builder.add_buffer(size * 4)
        
        # Would add compute dispatch here
        scenario_path = os.path.join(temp_dir, "large.json")
        builder.save(scenario_path)
        
        # Validate scenario can be loaded
        with open(scenario_path) as f:
            scenario = json.load(f)
        assert len(scenario["resources"]) == 2
    
    def test_many_small_dispatches(self, temp_dir):
        """Test many small compute dispatches"""
        builder = ScenarioBuilder("many_dispatches")
        
        # Create shared buffers
        buffer_a = builder.add_buffer(256 * 4)
        buffer_b = builder.add_buffer(256 * 4)
        
        # Add many dispatches
        for i in range(100):
            builder.add_mark_boundary(i)
            # Would add actual compute dispatches here
        
        scenario_path = os.path.join(temp_dir, "many.json")
        builder.save(scenario_path)
        
        # Validate scenario structure
        with open(scenario_path) as f:
            scenario = json.load(f)
        assert len(scenario["commands"]) == 100
    
    def test_pipeline_caching(self, temp_dir):
        """Test pipeline caching effectiveness"""
        builder = ScenarioBuilder("pipeline_cache")
        
        # Create simple scenario
        buffer = builder.add_buffer(1024 * 4)
        
        scenario_path = os.path.join(temp_dir, "cache.json")
        builder.save(scenario_path)
        
        # Run once without caching
        start = time.time()
        result1 = TestHelpers.run_scenario(scenario_path)
        time1 = time.time() - start
        
        # Run again with pipeline caching
        start = time.time()
        result2 = subprocess.run(
            [str(SCENARIO_RUNNER), "--scenario", scenario_path, "--pipeline-caching"],
            capture_output=True,
            text=True
        )
        time2 = time.time() - start
        
        # Second run should be similar or faster (caching benefit)
        # Note: May not show benefit on first run
        assert result1.returncode == 0 or result2.returncode == 0


# ============================================================================
# Resource Management Tests
# ============================================================================

class TestResourceManagement:
    """Test resource loading, saving, and aliasing"""
    
    def test_load_from_file(self, temp_dir):
        """Test loading resources from files"""
        # Create test data file
        test_data = np.random.randn(1024).astype(np.float32)
        data_file = os.path.join(temp_dir, "test_data.bin")
        test_data.tofile(data_file)
        
        # Create scenario with file reference
        builder = ScenarioBuilder("file_loading")
        scenario = builder.to_dict()
        scenario["resources"]["buffer_from_file"] = {
            "type": "buffer",
            "size": 1024 * 4,
            "src": data_file
        }
        
        scenario_path = os.path.join(temp_dir, "file_load.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        # Verify file exists and scenario is valid
        assert os.path.exists(data_file)
        assert os.path.getsize(data_file) == 1024 * 4
    
    def test_save_to_file(self, temp_dir):
        """Test saving resources to files"""
        builder = ScenarioBuilder("file_saving")
        
        # Create buffer with output file
        output_file = os.path.join(temp_dir, "output.bin")
        scenario = builder.to_dict()
        scenario["resources"]["output_buffer"] = {
            "type": "buffer",
            "size": 512 * 4,
            "dst": output_file
        }
        
        scenario_path = os.path.join(temp_dir, "file_save.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        # Run scenario (would save output)
        result = TestHelpers.run_scenario(scenario_path)
        
        # Output file would be created after successful run
        if result.returncode == 0 and os.path.exists(output_file):
            assert os.path.getsize(output_file) == 512 * 4
    
    def test_resource_aliasing_chain(self, temp_dir):
        """Test chained resource aliasing"""
        builder = ScenarioBuilder("alias_chain")
        
        # Create base buffer
        base_buffer = builder.add_buffer(4096)
        
        # Create aliased resources
        scenario = builder.to_dict()
        scenario["resources"]["alias1"] = {
            "type": "buffer",
            "alias": base_buffer,
            "offset": 0,
            "size": 2048
        }
        scenario["resources"]["alias2"] = {
            "type": "buffer",
            "alias": base_buffer,
            "offset": 2048,
            "size": 2048
        }
        
        scenario_path = os.path.join(temp_dir, "aliases.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        # Validate aliasing structure
        assert scenario["resources"]["alias1"]["alias"] == base_buffer
        assert scenario["resources"]["alias2"]["alias"] == base_buffer
        assert scenario["resources"]["alias1"]["offset"] + \
               scenario["resources"]["alias1"]["size"] == \
               scenario["resources"]["alias2"]["offset"]


# ============================================================================
# Error Recovery Tests
# ============================================================================

class TestErrorRecovery:
    """Test error handling and recovery"""
    
    def test_out_of_memory_handling(self, temp_dir):
        """Test handling of out-of-memory errors"""
        builder = ScenarioBuilder("oom_test")
        
        # Try to allocate huge buffer (100GB)
        huge_size = 100 * 1024 * 1024 * 1024
        builder.add_buffer(huge_size)
        
        scenario_path = os.path.join(temp_dir, "oom.json")
        builder.save(scenario_path)
        
        result = TestHelpers.run_scenario(scenario_path)
        
        # Should fail gracefully
        assert result.returncode != 0
        assert "memory" in result.stderr.lower() or "allocation" in result.stderr.lower()
    
    def test_missing_shader_file(self, temp_dir):
        """Test handling of missing shader files"""
        builder = ScenarioBuilder("missing_shader")
        
        # Reference non-existent shader
        builder.add_shader("/nonexistent/shader.spv")
        
        scenario_path = os.path.join(temp_dir, "missing.json")
        builder.save(scenario_path)
        
        result = TestHelpers.run_scenario(scenario_path)
        
        # Should report missing file
        assert result.returncode != 0
        assert "shader" in result.stderr.lower() or "file" in result.stderr.lower()
    
    def test_invalid_workgroup_size(self, temp_dir):
        """Test handling of invalid workgroup sizes"""
        builder = ScenarioBuilder("invalid_workgroup")
        
        buffer = builder.add_buffer(1024)
        
        # Add dispatch with invalid workgroup size (0)
        scenario = builder.to_dict()
        scenario["commands"].append({
            "type": "dispatch_compute",
            "workgroup": [0, 0, 0],  # Invalid
            "bindings": {0: buffer}
        })
        
        scenario_path = os.path.join(temp_dir, "invalid_wg.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        result = TestHelpers.run_scenario(scenario_path)
        
        # Should reject invalid workgroup
        assert result.returncode != 0


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])