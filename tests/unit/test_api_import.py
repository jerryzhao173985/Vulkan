#!/usr/bin/env python3
"""
Unit tests for vulkan_ml_sdk API import and basic functionality.

Tests that the vulkan_ml_sdk package can be imported correctly and
that core classes and functions are accessible.
"""

import os
import sys
import warnings
from pathlib import Path

import pytest

# Suppress deprecation warnings for version checks
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ============================================================================
# Path Setup
# ============================================================================

def _setup_sdk_path():
    """Add SDK Python package to sys.path if not already present."""
    # Find the SDK directory
    test_dir = Path(__file__).resolve().parent.parent
    repo_root = test_dir.parent
    sdk_python_path = repo_root / "builds" / "ARM-ML-SDK-Complete" / "lib" / "python"

    if sdk_python_path.exists() and str(sdk_python_path) not in sys.path:
        sys.path.insert(0, str(sdk_python_path))

    return sdk_python_path


SDK_PYTHON_PATH = _setup_sdk_path()


# ============================================================================
# Package Import Tests
# ============================================================================

class TestPackageImport:
    """Test that the vulkan_ml_sdk package can be imported."""

    def test_package_exists(self):
        """Test that SDK Python path exists."""
        assert SDK_PYTHON_PATH.exists(), f"SDK Python path not found: {SDK_PYTHON_PATH}"

    def test_import_vulkan_ml_sdk(self):
        """Test that vulkan_ml_sdk package can be imported."""
        import vulkan_ml_sdk
        assert vulkan_ml_sdk is not None

    def test_version_attribute(self):
        """Test that package has a version attribute."""
        import vulkan_ml_sdk
        assert hasattr(vulkan_ml_sdk, '__version__')
        assert vulkan_ml_sdk.__version__ == "0.1.0"

    def test_author_attribute(self):
        """Test that package has author information."""
        import vulkan_ml_sdk
        assert hasattr(vulkan_ml_sdk, '__author__')
        assert len(vulkan_ml_sdk.__author__) > 0

    def test_all_exports(self):
        """Test that __all__ exports are defined."""
        import vulkan_ml_sdk
        assert hasattr(vulkan_ml_sdk, '__all__')
        assert len(vulkan_ml_sdk.__all__) > 0


# ============================================================================
# Core Class Import Tests
# ============================================================================

class TestCoreClassImports:
    """Test that core classes can be imported from the package."""

    def test_import_sdk_class(self):
        """Test that SDK class can be imported."""
        from vulkan_ml_sdk import SDK
        assert SDK is not None

    def test_import_loaded_model(self):
        """Test that LoadedModel class can be imported."""
        from vulkan_ml_sdk import LoadedModel
        assert LoadedModel is not None

    def test_import_inference_engine(self):
        """Test that InferenceEngine class can be imported."""
        from vulkan_ml_sdk import InferenceEngine
        assert InferenceEngine is not None

    def test_import_inference_result(self):
        """Test that InferenceResult class can be imported."""
        from vulkan_ml_sdk import InferenceResult
        assert InferenceResult is not None

    def test_import_pipeline(self):
        """Test that Pipeline class can be imported."""
        from vulkan_ml_sdk import Pipeline
        assert Pipeline is not None

    def test_import_pipeline_stage(self):
        """Test that PipelineStage class can be imported."""
        from vulkan_ml_sdk import PipelineStage
        assert PipelineStage is not None


# ============================================================================
# Exception Class Import Tests
# ============================================================================

class TestExceptionImports:
    """Test that exception classes can be imported."""

    def test_import_sdk_error(self):
        """Test that SDKError exception can be imported."""
        from vulkan_ml_sdk import SDKError
        assert issubclass(SDKError, Exception)

    def test_import_scenario_runner_error(self):
        """Test that ScenarioRunnerError exception can be imported."""
        from vulkan_ml_sdk import ScenarioRunnerError
        assert issubclass(ScenarioRunnerError, Exception)

    def test_import_inference_error(self):
        """Test that InferenceError exception can be imported."""
        from vulkan_ml_sdk import InferenceError
        assert issubclass(InferenceError, Exception)

    def test_import_pipeline_error(self):
        """Test that PipelineError exception can be imported."""
        from vulkan_ml_sdk import PipelineError
        assert issubclass(PipelineError, Exception)


# ============================================================================
# Convenience Function Import Tests
# ============================================================================

class TestConvenienceFunctionImports:
    """Test that convenience functions can be imported."""

    def test_import_classify_function(self):
        """Test that classify function can be imported."""
        from vulkan_ml_sdk import classify
        assert callable(classify)

    def test_import_style_transfer_function(self):
        """Test that style_transfer function can be imported."""
        from vulkan_ml_sdk import style_transfer
        assert callable(style_transfer)

    def test_import_create_simple_pipeline(self):
        """Test that create_simple_pipeline function can be imported."""
        from vulkan_ml_sdk import create_simple_pipeline
        assert callable(create_simple_pipeline)


# ============================================================================
# Module Import Tests
# ============================================================================

class TestModuleImports:
    """Test that submodules can be imported."""

    def test_import_api_module(self):
        """Test that api module can be imported."""
        from vulkan_ml_sdk import api
        assert api is not None

    def test_import_models_module(self):
        """Test that models module can be imported."""
        from vulkan_ml_sdk import models
        assert models is not None
        assert hasattr(models, 'ModelRegistry')
        assert hasattr(models, 'ModelCache')

    def test_import_inference_module(self):
        """Test that inference module can be imported."""
        from vulkan_ml_sdk import inference
        assert inference is not None
        assert hasattr(inference, 'InferenceEngine')
        assert hasattr(inference, 'AsyncInference')

    def test_import_pipeline_module(self):
        """Test that pipeline module can be imported."""
        from vulkan_ml_sdk import pipeline
        assert pipeline is not None
        assert hasattr(pipeline, 'Pipeline')
        assert hasattr(pipeline, 'PipelineStage')

    def test_import_telemetry_module(self):
        """Test that telemetry module can be imported."""
        from vulkan_ml_sdk import telemetry
        assert telemetry is not None
        assert hasattr(telemetry, 'Telemetry')
        assert hasattr(telemetry, 'MetricType')


# ============================================================================
# Basic Class Instantiation Tests
# ============================================================================

class TestBasicInstantiation:
    """Test basic class instantiation without SDK dependencies."""

    def test_create_pipeline(self):
        """Test that Pipeline can be instantiated."""
        from vulkan_ml_sdk import Pipeline
        pipeline = Pipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'stages')
        assert len(pipeline.stages) == 0

    def test_create_pipeline_with_name(self):
        """Test Pipeline creation with a name."""
        from vulkan_ml_sdk import Pipeline
        pipeline = Pipeline(name="test_pipeline", description="A test pipeline")
        assert pipeline.name == "test_pipeline"
        assert pipeline.description == "A test pipeline"

    def test_create_pipeline_stage(self):
        """Test that PipelineStage can be instantiated."""
        from vulkan_ml_sdk import PipelineStage
        stage = PipelineStage(
            name="test_stage",
            model="test_model",
            inputs=["input_0"],
            outputs=["output_0"]
        )
        assert stage is not None
        assert stage.name == "test_stage"
        assert stage.model == "test_model"

    def test_create_telemetry(self):
        """Test that Telemetry can be instantiated."""
        from vulkan_ml_sdk.telemetry import Telemetry
        telemetry = Telemetry(name="test")
        assert telemetry is not None
        assert telemetry.name == "test"
        assert telemetry.supported_formats == ["json", "csv", "prometheus"]
        telemetry.close()

    def test_telemetry_record_latency(self):
        """Test recording latency metrics."""
        from vulkan_ml_sdk.telemetry import Telemetry
        telemetry = Telemetry()

        # Record some latency values
        telemetry.record_latency("test_op", 10.5)
        telemetry.record_latency("test_op", 12.0)
        telemetry.record_latency("test_op", 8.5)

        # Get statistics
        stats = telemetry.get_statistics("test_op")
        assert stats["count"] == 3
        assert stats["min"] == 8.5
        assert stats["max"] == 12.0

        telemetry.close()

    def test_telemetry_context_manager(self):
        """Test telemetry measure context manager."""
        from vulkan_ml_sdk.telemetry import Telemetry
        import time

        telemetry = Telemetry()

        with telemetry.measure("timing_test"):
            time.sleep(0.01)  # 10ms

        stats = telemetry.get_statistics("timing_test")
        assert stats["count"] == 1
        assert stats["mean"] >= 10.0  # At least 10ms

        telemetry.close()


# ============================================================================
# Models Module Tests
# ============================================================================

class TestModelsModule:
    """Test models module classes."""

    def test_model_registry_init(self):
        """Test ModelRegistry can be instantiated."""
        from vulkan_ml_sdk.models import ModelRegistry

        # Find SDK root
        test_dir = Path(__file__).resolve().parent.parent
        repo_root = test_dir.parent
        sdk_root = repo_root / "builds" / "ARM-ML-SDK-Complete"

        if sdk_root.exists():
            registry = ModelRegistry(sdk_root)
            assert registry is not None
            # Should find at least some models if SDK is properly set up
            models = registry.list_models()
            assert isinstance(models, list)

    def test_model_cache_init(self):
        """Test ModelCache can be instantiated."""
        from vulkan_ml_sdk.models import ModelCache
        import tempfile

        # Use temp directory for cache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir, max_size_mb=10)
            assert cache is not None
            assert cache.cache_dir == Path(tmpdir)
            assert cache.entry_count == 0

    def test_model_cache_stats(self):
        """Test ModelCache statistics."""
        from vulkan_ml_sdk.models import ModelCache
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)
            stats = cache.get_stats()

            assert "cache_dir" in stats
            assert "max_size_mb" in stats
            assert "current_size_mb" in stats
            assert "entry_count" in stats
            assert stats["entry_count"] == 0


# ============================================================================
# Inference Module Tests
# ============================================================================

class TestInferenceModule:
    """Test inference module classes."""

    def test_inference_result_creation(self):
        """Test InferenceResult can be created."""
        from vulkan_ml_sdk import InferenceResult

        result = InferenceResult(
            success=True,
            outputs={"output": [1, 2, 3]},
            execution_time_ms=25.5,
            model_name="test_model",
            metadata={"key": "value"}
        )

        assert result.success is True
        assert result.outputs == {"output": [1, 2, 3]}
        assert result.execution_time_ms == 25.5
        assert result.model_name == "test_model"

    def test_async_inference_class_exists(self):
        """Test that AsyncInference class is available."""
        from vulkan_ml_sdk.inference import AsyncInference
        assert AsyncInference is not None
        assert hasattr(AsyncInference, 'submit')


# ============================================================================
# Pipeline Module Tests
# ============================================================================

class TestPipelineModule:
    """Test pipeline module functionality."""

    def test_pipeline_add_stage(self):
        """Test adding stages to pipeline."""
        from vulkan_ml_sdk import Pipeline, PipelineStage

        pipeline = Pipeline(name="test")
        stage = PipelineStage(
            name="stage1",
            model="model1",
            inputs=["input"],
            outputs=["output"]
        )

        pipeline.add_stage(stage)

        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].name == "stage1"

    def test_pipeline_validation(self):
        """Test pipeline validation."""
        from vulkan_ml_sdk import Pipeline, PipelineStage

        pipeline = Pipeline()

        # Empty pipeline should be valid but return appropriate status
        result = pipeline.validate()
        assert isinstance(result, dict)
        assert "valid" in result

    def test_pipeline_has_optimize(self):
        """Test pipeline has optimize method."""
        from vulkan_ml_sdk import Pipeline

        pipeline = Pipeline()
        assert hasattr(pipeline, 'optimize')
        assert callable(pipeline.optimize)

    def test_create_simple_pipeline_function(self):
        """Test create_simple_pipeline function."""
        from vulkan_ml_sdk import create_simple_pipeline

        pipeline = create_simple_pipeline(
            name="simple_test",
            models=["model1", "model2"],
            auto_connect=True
        )

        assert pipeline is not None
        assert pipeline.name == "simple_test"
        assert len(pipeline.stages) == 2


# ============================================================================
# SDK Class Tests (without execution)
# ============================================================================

class TestSDKClass:
    """Test SDK class basic functionality."""

    def test_sdk_class_has_methods(self):
        """Test that SDK class has expected methods."""
        from vulkan_ml_sdk import SDK

        # Check key methods exist
        assert hasattr(SDK, 'load_model')
        assert hasattr(SDK, 'infer')
        assert hasattr(SDK, 'create_pipeline')
        assert hasattr(SDK, 'list_models')
        assert hasattr(SDK, 'list_shaders')
        assert hasattr(SDK, 'classify')
        assert hasattr(SDK, 'style_transfer')

    def test_sdk_style_models_defined(self):
        """Test that SDK has style transfer models defined."""
        from vulkan_ml_sdk import SDK

        assert hasattr(SDK, 'STYLE_TRANSFER_MODELS')
        styles = SDK.STYLE_TRANSFER_MODELS
        assert isinstance(styles, list)
        assert len(styles) > 0
        assert "la_muse" in styles


# ============================================================================
# Data Connection Tests
# ============================================================================

class TestDataConnection:
    """Test DataConnection class."""

    def test_data_connection_creation(self):
        """Test DataConnection can be created."""
        from vulkan_ml_sdk import DataConnection

        connection = DataConnection(
            source_stage="stage1",
            source_output="output",
            target_stage="stage2",
            target_input="input"
        )

        assert connection is not None
        assert connection.source_stage == "stage1"
        assert connection.target_stage == "stage2"


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
