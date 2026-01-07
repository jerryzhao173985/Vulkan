#!/usr/bin/env python3
"""
Vulkan ML SDK - Unified Python API for ML inference on Vulkan compute

This package provides a high-level Python interface for running machine learning
inference using Vulkan compute shaders on macOS ARM64 (Apple Silicon).

Features:
- Model loading and caching with warm-start support
- Synchronous and asynchronous inference
- Multi-model pipeline orchestration
- Performance telemetry and monitoring

Example usage:
    import vulkan_ml_sdk as vms

    sdk = vms.SDK()
    model = sdk.load_model("mobilenet_v2")
    result = model.infer(input_data)
"""

__version__ = "0.1.0"
__author__ = "ARM ML SDK Team"
__license__ = "MIT"

# Import core classes
from .api import SDK, SDKError, ScenarioRunnerError, LoadedModel, classify, style_transfer
from .inference import (
    InferenceEngine,
    InferenceResult,
    InferenceError,
    ModelLoadError,
    ExecutionError,
    InferenceTimeoutError,
    TimeoutError,  # Alias for backwards compatibility
)
from .pipeline import (
    Pipeline,
    PipelineStage,
    PipelineResult,
    PipelineError,
    DataConnection,
    create_simple_pipeline,
)

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "SDK",
    "SDKError",
    "ScenarioRunnerError",
    "LoadedModel",
    "InferenceEngine",
    "InferenceResult",
    "InferenceError",
    "ModelLoadError",
    "ExecutionError",
    "InferenceTimeoutError",
    "TimeoutError",  # Alias for backwards compatibility
    "Pipeline",
    "PipelineStage",
    "PipelineResult",
    "PipelineError",
    "DataConnection",
    "create_simple_pipeline",
    # Convenience functions for common ML tasks
    "classify",
    "style_transfer",
]
