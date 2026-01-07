#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
SPDX-License-Identifier: Apache-2.0

Integration Tests for Multi-Model Pipelines
Tests pipeline creation, stage management, data flow, optimization,
and end-to-end multi-model inference workflows.
"""

import pytest
import json
import tempfile
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add SDK to path - try multiple locations
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SDK_ROOT = Path("/Users/jerry/Vulkan")

# Try local builds first, then absolute path
possible_sdk_paths = [
    REPO_ROOT / "builds" / "ARM-ML-SDK-Complete",
    SDK_ROOT / "builds" / "ARM-ML-SDK-Complete",
    SDK_ROOT / ".worktrees" / "003-i-need-this-vulkan-directory-that-conrains-vartiou" / "builds" / "ARM-ML-SDK-Complete",
]

SDK_DIR = None
for sdk_path in possible_sdk_paths:
    if (sdk_path / "lib" / "python" / "vulkan_ml_sdk").exists():
        SDK_DIR = sdk_path
        break

if SDK_DIR is None:
    SDK_DIR = REPO_ROOT / "builds" / "ARM-ML-SDK-Complete"

PYTHON_LIB = SDK_DIR / "lib" / "python"
sys.path.insert(0, str(PYTHON_LIB))

# Import SDK modules
try:
    from vulkan_ml_sdk.pipeline import (
        Pipeline, PipelineStage, DataConnection,
        PipelineResult, StageResult, OptimizationConfig,
        PipelineError, StageNotFoundError, DataFlowError,
        PipelineValidationError, create_simple_pipeline
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# Constants
MODELS_DIR = SDK_DIR / "models"
SHADERS_DIR = SDK_DIR / "shaders"


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    temp_path = tempfile.mkdtemp(prefix="pipeline_test_")
    yield temp_path
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def basic_pipeline():
    """Create a basic two-stage pipeline"""
    pipeline = Pipeline(name="basic_test_pipeline")

    # Add preprocessing stage
    pipeline.add_stage(PipelineStage(
        name="preprocess",
        model="preprocessor",
        inputs=["raw_input"],
        outputs=["processed_data"]
    ))

    # Add inference stage
    pipeline.add_stage(PipelineStage(
        name="inference",
        model="mobilenet_v2",
        inputs=["processed_data"],
        outputs=["predictions"]
    ))

    # Connect stages
    pipeline.connect("preprocess", "processed_data", "inference", "processed_data")

    return pipeline


@pytest.fixture
def complex_pipeline():
    """Create a complex multi-branch pipeline"""
    pipeline = Pipeline(name="complex_test_pipeline")

    # Input preprocessing
    pipeline.add_stage(PipelineStage(
        name="normalize",
        model="normalizer",
        inputs=["image"],
        outputs=["normalized"]
    ))

    # Branch 1: Classification
    pipeline.add_stage(PipelineStage(
        name="classify",
        model="mobilenet_v2",
        inputs=["normalized"],
        outputs=["class_logits"]
    ))

    # Branch 2: Style transfer
    pipeline.add_stage(PipelineStage(
        name="style",
        model="la_muse",
        inputs=["normalized"],
        outputs=["stylized"]
    ))

    # Connect stages
    pipeline.connect("normalize", "normalized", "classify", "normalized")
    pipeline.connect("normalize", "normalized", "style", "normalized")

    return pipeline


@pytest.fixture
def mock_executor():
    """Create a mock executor for testing pipeline execution"""
    execution_log = []

    def executor(stage, inputs):
        execution_log.append({
            "stage": stage.name,
            "model": stage.model,
            "inputs": list(inputs.keys())
        })
        # Return mock outputs
        outputs = {}
        for output_name in stage.outputs:
            outputs[output_name] = f"mock_data_from_{stage.name}"
        return outputs

    return executor, execution_log


# ============================================================================
# Pipeline Creation Tests
# ============================================================================

@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK not available")
class TestPipelineCreation:
    """Test pipeline creation and basic configuration"""

    def test_create_empty_pipeline(self):
        """Test creating an empty pipeline"""
        pipeline = Pipeline(name="empty_pipeline")

        assert pipeline.name == "empty_pipeline"
        assert len(pipeline.stages) == 0
        assert len(pipeline.connections) == 0
        assert pipeline.execution_strategy == "sequential"

    def test_create_pipeline_with_auto_name(self):
        """Test creating pipeline with auto-generated name"""
        pipeline = Pipeline()

        assert pipeline.name.startswith("pipeline_")
        assert len(pipeline.name) > 10  # "pipeline_" + uuid

    def test_create_pipeline_with_description(self):
        """Test creating pipeline with description"""
        pipeline = Pipeline(
            name="described_pipeline",
            description="A test pipeline for classification"
        )

        assert pipeline.description == "A test pipeline for classification"

    def test_create_pipeline_parallel_strategy(self):
        """Test creating pipeline with parallel execution strategy"""
        pipeline = Pipeline(
            name="parallel_pipeline",
            execution_strategy="parallel"
        )

        assert pipeline.execution_strategy == "parallel"

    def test_invalid_execution_strategy(self):
        """Test that invalid execution strategy raises error"""
        pipeline = Pipeline(name="test")

        with pytest.raises(ValueError):
            pipeline.execution_strategy = "invalid_strategy"


# ============================================================================
# Stage Management Tests
# ============================================================================

@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK not available")
class TestStageManagement:
    """Test pipeline stage management"""

    def test_add_single_stage(self):
        """Test adding a single stage"""
        pipeline = Pipeline(name="single_stage")

        stage = PipelineStage(
            name="process",
            model="processor",
            inputs=["input"],
            outputs=["output"]
        )
        pipeline.add_stage(stage)

        assert len(pipeline.stages) == 1
        assert pipeline.has_stage("process")
        assert pipeline.get_stage("process") == stage

    def test_add_multiple_stages(self):
        """Test adding multiple stages"""
        pipeline = Pipeline(name="multi_stage")

        for i in range(5):
            pipeline.add_stage(PipelineStage(
                name=f"stage_{i}",
                model=f"model_{i}",
                inputs=[f"input_{i}"],
                outputs=[f"output_{i}"]
            ))

        assert len(pipeline.stages) == 5
        for i in range(5):
            assert pipeline.has_stage(f"stage_{i}")

    def test_add_duplicate_stage_raises_error(self):
        """Test that adding duplicate stage name raises error"""
        pipeline = Pipeline(name="duplicate_test")

        pipeline.add_stage(PipelineStage(
            name="stage",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))

        with pytest.raises(PipelineError):
            pipeline.add_stage(PipelineStage(
                name="stage",  # Duplicate name
                model="other_model",
                inputs=["in2"],
                outputs=["out2"]
            ))

    def test_remove_stage(self):
        """Test removing a stage"""
        pipeline = Pipeline(name="remove_test")

        pipeline.add_stage(PipelineStage(
            name="to_remove",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))

        assert pipeline.has_stage("to_remove")

        result = pipeline.remove_stage("to_remove")

        assert result is True
        assert not pipeline.has_stage("to_remove")

    def test_remove_nonexistent_stage(self):
        """Test removing a stage that doesn't exist"""
        pipeline = Pipeline(name="remove_test")

        result = pipeline.remove_stage("nonexistent")

        assert result is False

    def test_get_nonexistent_stage_raises_error(self):
        """Test getting a stage that doesn't exist raises error"""
        pipeline = Pipeline(name="get_test")

        with pytest.raises(StageNotFoundError):
            pipeline.get_stage("nonexistent")

    def test_stage_with_config(self):
        """Test stage with additional configuration"""
        pipeline = Pipeline(name="config_test")

        stage = PipelineStage(
            name="configured",
            model="model",
            inputs=["in"],
            outputs=["out"],
            config={
                "batch_size": 32,
                "precision": "fp16",
                "optimization": "enabled"
            }
        )
        pipeline.add_stage(stage)

        retrieved = pipeline.get_stage("configured")
        assert retrieved.config["batch_size"] == 32
        assert retrieved.config["precision"] == "fp16"

    def test_disable_stage(self):
        """Test disabling a stage"""
        pipeline = Pipeline(name="disable_test")

        stage = PipelineStage(
            name="disabled",
            model="model",
            inputs=["in"],
            outputs=["out"],
            enabled=False
        )
        pipeline.add_stage(stage)

        assert pipeline.get_stage("disabled").enabled is False


# ============================================================================
# Data Connection Tests
# ============================================================================

@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK not available")
class TestDataConnections:
    """Test data flow connections between stages"""

    def test_connect_stages(self):
        """Test connecting two stages"""
        pipeline = Pipeline(name="connect_test")

        pipeline.add_stage(PipelineStage(
            name="source",
            model="src_model",
            inputs=["in"],
            outputs=["out"]
        ))

        pipeline.add_stage(PipelineStage(
            name="target",
            model="tgt_model",
            inputs=["in"],
            outputs=["out"]
        ))

        pipeline.connect("source", "out", "target", "in")

        assert len(pipeline.connections) == 1
        conn = pipeline.connections[0]
        assert conn.source_stage == "source"
        assert conn.source_output == "out"
        assert conn.target_stage == "target"
        assert conn.target_input == "in"

    def test_connect_nonexistent_source_raises_error(self):
        """Test connecting from nonexistent source raises error"""
        pipeline = Pipeline(name="error_test")

        pipeline.add_stage(PipelineStage(
            name="target",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))

        with pytest.raises(StageNotFoundError):
            pipeline.connect("nonexistent", "out", "target", "in")

    def test_connect_nonexistent_target_raises_error(self):
        """Test connecting to nonexistent target raises error"""
        pipeline = Pipeline(name="error_test")

        pipeline.add_stage(PipelineStage(
            name="source",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))

        with pytest.raises(StageNotFoundError):
            pipeline.connect("source", "out", "nonexistent", "in")

    def test_connect_invalid_output_raises_error(self):
        """Test connecting invalid output raises error"""
        pipeline = Pipeline(name="error_test")

        pipeline.add_stage(PipelineStage(
            name="source",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))
        pipeline.add_stage(PipelineStage(
            name="target",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))

        with pytest.raises(DataFlowError):
            pipeline.connect("source", "invalid_output", "target", "in")

    def test_connect_invalid_input_raises_error(self):
        """Test connecting invalid input raises error"""
        pipeline = Pipeline(name="error_test")

        pipeline.add_stage(PipelineStage(
            name="source",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))
        pipeline.add_stage(PipelineStage(
            name="target",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))

        with pytest.raises(DataFlowError):
            pipeline.connect("source", "out", "target", "invalid_input")

    def test_duplicate_connection_raises_error(self):
        """Test duplicate connection to same input raises error"""
        pipeline = Pipeline(name="duplicate_test")

        pipeline.add_stage(PipelineStage(
            name="source1",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))
        pipeline.add_stage(PipelineStage(
            name="source2",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))
        pipeline.add_stage(PipelineStage(
            name="target",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))

        pipeline.connect("source1", "out", "target", "in")

        with pytest.raises(DataFlowError):
            pipeline.connect("source2", "out", "target", "in")

    def test_disconnect_stages(self):
        """Test disconnecting stages"""
        pipeline = Pipeline(name="disconnect_test")

        pipeline.add_stage(PipelineStage(
            name="source",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))
        pipeline.add_stage(PipelineStage(
            name="target",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))

        pipeline.connect("source", "out", "target", "in")
        assert len(pipeline.connections) == 1

        result = pipeline.disconnect("source", "out", "target", "in")

        assert result is True
        assert len(pipeline.connections) == 0

    def test_connection_with_transform(self):
        """Test connection with data transformation"""
        pipeline = Pipeline(name="transform_test")

        pipeline.add_stage(PipelineStage(
            name="source",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))
        pipeline.add_stage(PipelineStage(
            name="target",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))

        pipeline.connect("source", "out", "target", "in", transform="reshape")

        conn = pipeline.connections[0]
        assert conn.transform == "reshape"


# ============================================================================
# Pipeline Validation Tests
# ============================================================================

@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK not available")
class TestPipelineValidation:
    """Test pipeline validation"""

    def test_validate_valid_pipeline(self, basic_pipeline):
        """Test validating a valid pipeline"""
        result = basic_pipeline.validate()

        assert result["valid"] is True
        assert result["stage_count"] == 2
        assert result["connection_count"] == 1
        assert len(result["errors"]) == 0

    def test_validate_empty_pipeline(self):
        """Test validating an empty pipeline"""
        pipeline = Pipeline(name="empty")

        result = pipeline.validate()

        assert result["valid"] is True
        assert "Pipeline has no stages" in result["warnings"]

    def test_validate_missing_model_raises_error(self):
        """Test that stage without model fails validation"""
        pipeline = Pipeline(name="no_model")

        pipeline.add_stage(PipelineStage(
            name="stage",
            model="",  # Empty model
            inputs=["in"],
            outputs=["out"]
        ))

        with pytest.raises(PipelineValidationError):
            pipeline.validate()

    def test_validate_unconnected_input_warning(self):
        """Test that unconnected inputs generate warnings"""
        pipeline = Pipeline(name="unconnected")

        pipeline.add_stage(PipelineStage(
            name="stage",
            model="model",
            inputs=["external_input"],
            outputs=["out"]
        ))

        result = pipeline.validate()

        assert result["valid"] is True
        assert any("external_input" in w for w in result["warnings"])

    def test_validate_circular_dependency(self):
        """Test that circular dependencies fail validation"""
        pipeline = Pipeline(name="circular")

        # Create stages that could form a cycle
        pipeline.add_stage(PipelineStage(
            name="a",
            model="model",
            inputs=["from_c"],
            outputs=["to_b"]
        ))
        pipeline.add_stage(PipelineStage(
            name="b",
            model="model",
            inputs=["to_b"],
            outputs=["to_c"]
        ))
        pipeline.add_stage(PipelineStage(
            name="c",
            model="model",
            inputs=["to_c"],
            outputs=["from_c"]
        ))

        # Connect in a cycle: a -> b -> c -> a
        pipeline.connect("a", "to_b", "b", "to_b")
        pipeline.connect("b", "to_c", "c", "to_c")
        pipeline.connect("c", "from_c", "a", "from_c")

        with pytest.raises(PipelineValidationError):
            pipeline.validate()


# ============================================================================
# Pipeline Execution Tests
# ============================================================================

@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK not available")
class TestPipelineExecution:
    """Test pipeline execution"""

    def test_execute_basic_pipeline(self, basic_pipeline, mock_executor):
        """Test executing a basic pipeline"""
        executor, log = mock_executor
        basic_pipeline.set_executor(executor)

        result = basic_pipeline.execute({"raw_input": "test_data"})

        assert result.success is True
        assert result.pipeline_name == "basic_test_pipeline"
        assert len(result.stage_results) == 2
        assert result.total_time_ms > 0

    def test_execute_with_inputs(self, basic_pipeline, mock_executor):
        """Test executing pipeline with external inputs"""
        executor, log = mock_executor
        basic_pipeline.set_executor(executor)

        inputs = {
            "raw_input": {"data": [1, 2, 3], "shape": [1, 3]}
        }

        result = basic_pipeline.execute(inputs)

        assert result.success is True
        assert log[0]["stage"] == "preprocess"
        assert "raw_input" in log[0]["inputs"]

    def test_execution_order(self, basic_pipeline):
        """Test that stages execute in correct order"""
        order = basic_pipeline.get_execution_order()

        # Preprocess should come before inference
        preprocess_idx = order.index("preprocess")
        inference_idx = order.index("inference")

        assert preprocess_idx < inference_idx

    def test_execute_disabled_stage_skipped(self, mock_executor):
        """Test that disabled stages are skipped"""
        pipeline = Pipeline(name="disabled_test")

        pipeline.add_stage(PipelineStage(
            name="enabled",
            model="model",
            inputs=["in"],
            outputs=["out"],
            enabled=True
        ))
        pipeline.add_stage(PipelineStage(
            name="disabled",
            model="model",
            inputs=["in"],
            outputs=["out"],
            enabled=False
        ))

        executor, log = mock_executor
        pipeline.set_executor(executor)

        result = pipeline.execute({"in": "data"})

        # Only enabled stage should execute
        assert len(log) == 1
        assert log[0]["stage"] == "enabled"

    def test_execute_failure_stops_sequential(self, basic_pipeline):
        """Test that failure stops sequential execution"""
        def failing_executor(stage, inputs):
            if stage.name == "preprocess":
                raise RuntimeError("Simulated failure")
            return {"output": "data"}

        basic_pipeline.set_executor(failing_executor)

        result = basic_pipeline.execute({"raw_input": "data"})

        assert result.success is False
        assert len(result.stage_results) == 1  # Only preprocess tried
        assert result.stage_results[0].error is not None

    def test_execute_returns_final_outputs(self, mock_executor):
        """Test that execute returns final stage outputs"""
        pipeline = Pipeline(name="output_test")

        pipeline.add_stage(PipelineStage(
            name="final",
            model="model",
            inputs=["in"],
            outputs=["result"]
        ))

        executor, log = mock_executor
        pipeline.set_executor(executor)

        result = pipeline.execute({"in": "data"})

        assert "result" in result.final_outputs


# ============================================================================
# Pipeline Optimization Tests
# ============================================================================

@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK not available")
class TestPipelineOptimization:
    """Test pipeline optimization features"""

    def test_optimize_with_quantization(self, basic_pipeline):
        """Test applying quantization optimization"""
        config = OptimizationConfig(
            enable_quantization=True,
            quantization_bits=8
        )

        result = basic_pipeline.optimize(config)

        assert result.success is True
        assert "quantization_8bit" in result.optimizations_applied
        assert len(result.quantized_stages) == 2
        assert result.memory_reduction < 1.0  # Memory reduced

    def test_optimize_with_fp16(self, basic_pipeline):
        """Test applying FP16 optimization"""
        config = OptimizationConfig(
            enable_quantization=False,
            use_fp16=True
        )

        result = basic_pipeline.optimize(config)

        assert result.success is True
        assert "fp16_acceleration" in result.optimizations_applied
        assert result.estimated_speedup > 1.0

    def test_optimize_with_fusion(self, basic_pipeline):
        """Test applying operator fusion"""
        config = OptimizationConfig(
            enable_fusion=True
        )

        result = basic_pipeline.optimize(config)

        assert result.success is True
        assert "operator_fusion" in result.optimizations_applied

    def test_enable_quantization_convenience(self, basic_pipeline):
        """Test convenience method for enabling quantization"""
        result = basic_pipeline.enable_quantization(bits=8)

        assert result is basic_pipeline  # Method chaining

        report = basic_pipeline.get_optimization_report()
        assert report["optimization_config"]["enable_quantization"] is True

    def test_get_optimization_report(self, basic_pipeline):
        """Test getting optimization report"""
        basic_pipeline.optimize(OptimizationConfig(
            enable_quantization=True,
            use_fp16=True
        ))

        report = basic_pipeline.get_optimization_report()

        assert report["pipeline_name"] == "basic_test_pipeline"
        assert len(report["stage_details"]) == 2


# ============================================================================
# Pipeline Serialization Tests
# ============================================================================

@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK not available")
class TestPipelineSerialization:
    """Test pipeline serialization and deserialization"""

    def test_to_dict(self, basic_pipeline):
        """Test converting pipeline to dictionary"""
        data = basic_pipeline.to_dict()

        assert data["name"] == "basic_test_pipeline"
        assert len(data["stages"]) == 2
        assert len(data["connections"]) == 1
        assert "execution_strategy" in data

    def test_from_dict(self, basic_pipeline):
        """Test creating pipeline from dictionary"""
        data = basic_pipeline.to_dict()

        restored = Pipeline.from_dict(data)

        assert restored.name == basic_pipeline.name
        assert len(restored.stages) == len(basic_pipeline.stages)
        assert len(restored.connections) == len(basic_pipeline.connections)

    def test_to_json(self, basic_pipeline):
        """Test serializing pipeline to JSON"""
        json_str = basic_pipeline.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["name"] == "basic_test_pipeline"

    def test_from_json(self, basic_pipeline):
        """Test deserializing pipeline from JSON"""
        json_str = basic_pipeline.to_json()

        restored = Pipeline.from_json(json_str)

        assert restored.name == basic_pipeline.name

    def test_save_and_load(self, basic_pipeline, temp_dir):
        """Test saving and loading pipeline from file"""
        filepath = Path(temp_dir) / "pipeline.json"

        basic_pipeline.save(filepath)
        assert filepath.exists()

        loaded = Pipeline.load(filepath)

        assert loaded.name == basic_pipeline.name
        assert len(loaded.stages) == 2


# ============================================================================
# Multi-Model Integration Tests
# ============================================================================

@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK not available")
class TestMultiModelIntegration:
    """Test integration with real SDK models"""

    def test_create_style_transfer_pipeline(self):
        """Test creating a style transfer pipeline with real models"""
        pipeline = Pipeline(
            name="style_transfer_pipeline",
            description="Multi-style transfer with model switching"
        )

        # Input preprocessing
        pipeline.add_stage(PipelineStage(
            name="preprocess",
            model="image_preprocessor",
            inputs=["image"],
            outputs=["normalized"]
        ))

        # Add multiple style models
        style_models = ["la_muse", "udnie", "mirror"]
        for model in style_models:
            model_path = MODELS_DIR / f"{model}.tflite"
            if model_path.exists():
                pipeline.add_stage(PipelineStage(
                    name=f"style_{model}",
                    model=model,
                    inputs=["normalized"],
                    outputs=[f"{model}_output"]
                ))
                pipeline.connect("preprocess", "normalized", f"style_{model}", "normalized")

        # Validate pipeline
        result = pipeline.validate()
        assert result["valid"] is True

    def test_create_detection_classification_pipeline(self):
        """Test creating detection + classification pipeline"""
        pipeline = Pipeline(name="detect_classify_pipeline")

        # Detection stage
        pipeline.add_stage(PipelineStage(
            name="detect",
            model="fire_detection",
            inputs=["image"],
            outputs=["detection", "regions"]
        ))

        # Classification stage
        pipeline.add_stage(PipelineStage(
            name="classify",
            model="mobilenet_v2",
            inputs=["regions"],
            outputs=["classifications"]
        ))

        # Connect detection to classification
        pipeline.connect("detect", "regions", "classify", "regions")

        result = pipeline.validate()
        assert result["valid"] is True

    def test_list_available_models(self):
        """Test listing available models for pipeline stages"""
        if not MODELS_DIR.exists():
            pytest.skip("Models directory not found")

        models = list(MODELS_DIR.glob("*.tflite"))

        assert len(models) > 0

        # Can create stage for each model
        pipeline = Pipeline(name="all_models")
        for model_path in models:
            pipeline.add_stage(PipelineStage(
                name=f"stage_{model_path.stem}",
                model=model_path.stem,
                inputs=["input"],
                outputs=["output"]
            ))

        assert len(pipeline.stages) == len(models)

    def test_create_simple_pipeline_helper(self):
        """Test create_simple_pipeline helper function"""
        models = ["preprocessor", "mobilenet_v2", "postprocessor"]

        pipeline = create_simple_pipeline("simple_inference", models)

        assert pipeline.name == "simple_inference"
        assert len(pipeline.stages) == 3
        # Auto-connections should be created
        assert len(pipeline.connections) == 2


# ============================================================================
# Pipeline Helpers and Utilities Tests
# ============================================================================

@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK not available")
class TestPipelineUtilities:
    """Test pipeline utility methods"""

    def test_pipeline_length(self, basic_pipeline):
        """Test pipeline length"""
        assert len(basic_pipeline) == 2

    def test_pipeline_contains(self, basic_pipeline):
        """Test stage membership check"""
        assert "preprocess" in basic_pipeline
        assert "nonexistent" not in basic_pipeline

    def test_pipeline_iteration(self, basic_pipeline):
        """Test iterating over pipeline stages"""
        stage_names = [s.name for s in basic_pipeline]

        assert "preprocess" in stage_names
        assert "inference" in stage_names

    def test_pipeline_clear(self, basic_pipeline):
        """Test clearing pipeline"""
        basic_pipeline.clear()

        assert len(basic_pipeline.stages) == 0
        assert len(basic_pipeline.connections) == 0

    def test_pipeline_repr(self, basic_pipeline):
        """Test pipeline string representation"""
        repr_str = repr(basic_pipeline)

        assert "Pipeline" in repr_str
        assert "basic_test_pipeline" in repr_str
        assert "stages=2" in repr_str


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
