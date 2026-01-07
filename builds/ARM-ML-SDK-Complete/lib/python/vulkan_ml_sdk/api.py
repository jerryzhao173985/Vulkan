#!/usr/bin/env python3
"""
Core SDK class with environment setup and scenario-runner wrapper.

This module provides the main SDK class for interacting with the ARM ML SDK
for Vulkan compute on macOS ARM64 (Apple Silicon).
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .inference import InferenceEngine, InferenceResult
    from .pipeline import Pipeline, PipelineStage


class SDKError(Exception):
    """Base exception for SDK errors."""
    pass


class ScenarioRunnerError(SDKError):
    """Error during scenario-runner execution."""
    pass


class LoadedModel:
    """
    Represents a loaded model with inference capabilities.

    Provides a convenient interface for running inference on a specific model
    that has been loaded by the SDK.

    Example:
        sdk = SDK()
        model = sdk.load_model("mobilenet_v2")
        result = model.infer(input_data)
    """

    def __init__(self, name: str, info: Dict[str, Any], engine: "InferenceEngine"):
        """
        Initialize LoadedModel.

        Args:
            name: Model name.
            info: Model information dictionary.
            engine: Reference to the InferenceEngine.
        """
        self._name = name
        self._info = info
        self._engine = engine

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def path(self) -> str:
        """Path to the model file."""
        return self._info.get("path", "")

    @property
    def size(self) -> int:
        """Model file size in bytes."""
        return self._info.get("size", 0)

    @property
    def info(self) -> Dict[str, Any]:
        """Full model information dictionary."""
        return self._info.copy()

    def infer(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 300000,
        profiling: bool = False
    ) -> "InferenceResult":
        """
        Run inference on this model.

        Args:
            inputs: Input data dictionary.
            timeout_ms: Maximum execution time in milliseconds.
            profiling: If True, enable performance profiling.

        Returns:
            InferenceResult with inference results.
        """
        return self._engine.infer(
            model_name=self._name,
            inputs=inputs,
            timeout_ms=timeout_ms,
            profiling=profiling
        )

    def __repr__(self) -> str:
        return f"LoadedModel(name={self._name!r}, size={self.size})"


class SDK:
    """
    Main SDK class for Vulkan ML inference.

    Provides environment setup, path management, and scenario-runner wrapper
    for running ML inference on Vulkan compute shaders.

    Example:
        sdk = SDK()
        print(f"SDK root: {sdk.sdk_root}")
        print(f"Available models: {sdk.list_models()}")

        # Run inference
        result = sdk.run_scenario("test.json", output_dir="results/")
    """

    def __init__(self, sdk_root: Optional[str] = None):
        """
        Initialize the SDK with environment setup.

        Args:
            sdk_root: Optional path to SDK root. If not provided, auto-detects
                      from package location.
        """
        self._sdk_root = self._resolve_sdk_root(sdk_root)
        self._setup_environment()
        self._validate_installation()
        self._inference_engine = None
        self._loaded_models: Dict[str, LoadedModel] = {}

    def _resolve_sdk_root(self, sdk_root: Optional[str]) -> Path:
        """
        Resolve the SDK root directory.

        Args:
            sdk_root: User-provided SDK root or None for auto-detection.

        Returns:
            Path to SDK root directory.
        """
        if sdk_root:
            return Path(sdk_root).resolve()

        # Auto-detect: lib/python/vulkan_ml_sdk/api.py -> lib/python -> lib -> SDK_ROOT
        current_file = Path(__file__).resolve()
        return current_file.parent.parent.parent.parent

    def _setup_environment(self) -> None:
        """Configure environment variables for Vulkan and SDK libraries."""
        lib_path = str(self.lib_dir)
        system_lib = "/usr/local/lib"

        # Set DYLD_LIBRARY_PATH for macOS
        current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
        paths_to_add = [system_lib, lib_path]

        # Build new path, avoiding duplicates
        existing_paths = current_dyld.split(":") if current_dyld else []
        new_paths = []
        for path in paths_to_add + existing_paths:
            if path and path not in new_paths:
                new_paths.append(path)

        os.environ["DYLD_LIBRARY_PATH"] = ":".join(new_paths)

        # Store the environment for subprocess calls
        self._env = os.environ.copy()

    def _validate_installation(self) -> None:
        """Validate that required SDK components are present."""
        # Check scenario-runner exists
        if not self.scenario_runner_path.exists():
            raise SDKError(
                f"scenario-runner not found at {self.scenario_runner_path}. "
                "Please verify SDK installation."
            )

    @property
    def sdk_root(self) -> Path:
        """Path to the SDK root directory."""
        return self._sdk_root

    @property
    def bin_dir(self) -> Path:
        """Path to the SDK bin directory."""
        return self._sdk_root / "bin"

    @property
    def lib_dir(self) -> Path:
        """Path to the SDK lib directory."""
        return self._sdk_root / "lib"

    @property
    def models_dir(self) -> Path:
        """Path to the SDK models directory."""
        return self._sdk_root / "models"

    @property
    def shaders_dir(self) -> Path:
        """Path to the SDK shaders directory."""
        return self._sdk_root / "shaders"

    @property
    def tools_dir(self) -> Path:
        """Path to the SDK tools directory."""
        return self._sdk_root / "tools"

    @property
    def scenario_runner_path(self) -> Path:
        """Path to the scenario-runner executable."""
        return self.bin_dir / "scenario-runner"

    def get_version(self) -> Optional[str]:
        """
        Get the scenario-runner version.

        Returns:
            Version string or None if unable to retrieve.
        """
        try:
            result = subprocess.run(
                [str(self.scenario_runner_path), "--version"],
                capture_output=True,
                text=True,
                env=self._env,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return result.stderr.strip() if result.stderr else None
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None

    def list_models(self) -> List[str]:
        """
        List available TFLite models.

        Returns:
            List of model names (without .tflite extension).
        """
        if not self.models_dir.exists():
            return []

        models = []
        for model_file in self.models_dir.glob("*.tflite"):
            models.append(model_file.stem)
        return sorted(models)

    def list_shaders(self) -> List[str]:
        """
        List available SPIR-V compute shaders.

        Returns:
            List of shader names (without .spv extension).
        """
        if not self.shaders_dir.exists():
            return []

        shaders = []
        for shader_file in self.shaders_dir.glob("*.spv"):
            shaders.append(shader_file.stem)
        return sorted(shaders)

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get the full path to a model file.

        Args:
            model_name: Model name (with or without .tflite extension).

        Returns:
            Path to the model file, or None if not found.
        """
        if not model_name.endswith(".tflite"):
            model_name = f"{model_name}.tflite"

        model_path = self.models_dir / model_name
        return model_path if model_path.exists() else None

    def get_shader_path(self, shader_name: str) -> Optional[Path]:
        """
        Get the full path to a shader file.

        Args:
            shader_name: Shader name (with or without .spv extension).

        Returns:
            Path to the shader file, or None if not found.
        """
        if not shader_name.endswith(".spv"):
            shader_name = f"{shader_name}.spv"

        shader_path = self.shaders_dir / shader_name
        return shader_path if shader_path.exists() else None

    def run_scenario(
        self,
        scenario: Union[str, Path, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
        dry_run: bool = False,
        profiling_dump_path: Optional[Union[str, Path]] = None,
        pipeline_caching: bool = False,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run a scenario through the scenario-runner.

        Args:
            scenario: Path to scenario JSON file or scenario dict.
            output_dir: Directory for output files.
            dry_run: If True, validate scenario without executing.
            profiling_dump_path: Path for performance metrics output.
            pipeline_caching: If True, enable shader pipeline caching.
            timeout: Maximum execution time in seconds.

        Returns:
            Dict with execution results including returncode, stdout, stderr.

        Raises:
            ScenarioRunnerError: If scenario execution fails.
        """
        # Handle scenario as dict - write to temp file
        temp_scenario_file = None
        if isinstance(scenario, dict):
            import tempfile
            fd, temp_scenario_file = tempfile.mkstemp(suffix=".json")
            with os.fdopen(fd, 'w') as f:
                json.dump(scenario, f, indent=2)
            scenario_path = temp_scenario_file
        else:
            scenario_path = str(scenario)

        try:
            # Build command
            cmd = [str(self.scenario_runner_path), "--scenario", scenario_path]

            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                cmd.extend(["--output", str(output_path)])

            if dry_run:
                cmd.append("--dry-run")

            if profiling_dump_path:
                cmd.extend(["--profiling-dump-path", str(profiling_dump_path)])

            if pipeline_caching:
                cmd.append("--pipeline-caching")

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._env,
                timeout=timeout
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }

        except subprocess.TimeoutExpired as e:
            raise ScenarioRunnerError(f"Scenario execution timed out after {timeout}s")
        except Exception as e:
            raise ScenarioRunnerError(f"Failed to run scenario: {e}")
        finally:
            # Clean up temp file
            if temp_scenario_file and os.path.exists(temp_scenario_file):
                os.unlink(temp_scenario_file)

    def validate_scenario(self, scenario: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a scenario without executing it.

        Args:
            scenario: Path to scenario JSON file or scenario dict.

        Returns:
            Validation result dict.
        """
        return self.run_scenario(scenario, dry_run=True)

    def get_info(self) -> Dict[str, Any]:
        """
        Get SDK information summary.

        Returns:
            Dict with SDK paths, version, and available resources.
        """
        return {
            "sdk_root": str(self.sdk_root),
            "version": self.get_version(),
            "paths": {
                "bin": str(self.bin_dir),
                "lib": str(self.lib_dir),
                "models": str(self.models_dir),
                "shaders": str(self.shaders_dir),
                "tools": str(self.tools_dir),
            },
            "resources": {
                "models": self.list_models(),
                "shaders": self.list_shaders(),
                "model_count": len(self.list_models()),
                "shader_count": len(self.list_shaders()),
            },
            "environment": {
                "DYLD_LIBRARY_PATH": os.environ.get("DYLD_LIBRARY_PATH", ""),
            }
        }

    # =========================================================================
    # Model Loading and Inference Methods
    # =========================================================================

    def _get_inference_engine(self) -> "InferenceEngine":
        """
        Get or create the InferenceEngine instance.

        Returns:
            InferenceEngine instance.
        """
        if self._inference_engine is None:
            from .inference import InferenceEngine
            self._inference_engine = InferenceEngine(self._sdk_root)
        return self._inference_engine

    def load_model(self, model_name: str) -> LoadedModel:
        """
        Load a model for inference.

        Loads the specified model and returns a LoadedModel object that can
        be used to run inference.

        Args:
            model_name: Name of the model to load (with or without .tflite extension).

        Returns:
            LoadedModel object for running inference.

        Raises:
            SDKError: If model cannot be loaded.

        Example:
            sdk = SDK()
            model = sdk.load_model("mobilenet_v2")
            result = model.infer({"input": data})
        """
        # Check if already loaded
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        engine = self._get_inference_engine()

        try:
            model_info = engine.load_model(model_name)
            loaded_model = LoadedModel(model_name, model_info, engine)
            self._loaded_models[model_name] = loaded_model
            return loaded_model
        except Exception as e:
            raise SDKError(f"Failed to load model '{model_name}': {e}")

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_name: Name of the model to unload.

        Returns:
            True if model was unloaded, False if not found.
        """
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            engine = self._get_inference_engine()
            engine.unload_model(model_name)
            return True
        return False

    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if model is loaded, False otherwise.
        """
        return model_name in self._loaded_models

    def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded model names.

        Returns:
            List of loaded model names.
        """
        return list(self._loaded_models.keys())

    def infer(
        self,
        model_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 300000,
        profiling: bool = False
    ) -> "InferenceResult":
        """
        Run inference on a model.

        This is a convenience method that loads the model if needed and runs
        inference. For repeated inference on the same model, prefer using
        load_model() and calling infer() on the returned LoadedModel.

        Args:
            model_name: Name of the model to use.
            inputs: Input data dictionary.
            timeout_ms: Maximum execution time in milliseconds.
            profiling: If True, enable performance profiling.

        Returns:
            InferenceResult with inference results.

        Raises:
            SDKError: If inference fails.

        Example:
            sdk = SDK()
            result = sdk.infer("mobilenet_v2", {"input": image_data})
        """
        engine = self._get_inference_engine()
        return engine.infer(
            model_name=model_name,
            inputs=inputs,
            timeout_ms=timeout_ms,
            profiling=profiling
        )

    # =========================================================================
    # Pipeline Methods
    # =========================================================================

    def create_pipeline(
        self,
        name: Optional[str] = None,
        description: str = "",
        execution_strategy: str = "sequential"
    ) -> "Pipeline":
        """
        Create a new inference pipeline.

        Creates an empty pipeline that can be populated with stages
        for multi-model inference.

        Args:
            name: Name for the pipeline. Auto-generated if not provided.
            description: Human-readable description of the pipeline.
            execution_strategy: "sequential" or "parallel".

        Returns:
            New Pipeline instance.

        Example:
            sdk = SDK()
            pipeline = sdk.create_pipeline("image_classifier")
            pipeline.add_stage(PipelineStage(
                name="classify",
                model="mobilenet_v2",
                inputs=["image"],
                outputs=["predictions"]
            ))
        """
        from .pipeline import Pipeline
        return Pipeline(
            name=name,
            description=description,
            execution_strategy=execution_strategy
        )

    def create_simple_pipeline(
        self,
        name: str,
        models: List[str],
        auto_connect: bool = True
    ) -> "Pipeline":
        """
        Create a simple sequential pipeline from a list of models.

        This is a convenience method that creates a pipeline with one
        stage per model, automatically connecting them in sequence.

        Args:
            name: Name for the pipeline.
            models: List of model names in execution order.
            auto_connect: If True, automatically connect stages.

        Returns:
            Configured Pipeline instance.

        Example:
            sdk = SDK()
            pipeline = sdk.create_simple_pipeline(
                "style_transfer",
                ["preprocessor", "la_muse", "postprocessor"]
            )
        """
        from .pipeline import create_simple_pipeline
        return create_simple_pipeline(name, models, auto_connect)

    def load_pipeline(self, path: Union[str, Path]) -> "Pipeline":
        """
        Load a pipeline from a JSON file.

        Args:
            path: Path to the pipeline JSON file.

        Returns:
            Pipeline instance.

        Raises:
            SDKError: If pipeline cannot be loaded.

        Example:
            sdk = SDK()
            pipeline = sdk.load_pipeline("pipelines/my_pipeline.json")
        """
        from .pipeline import Pipeline
        try:
            return Pipeline.load(path)
        except Exception as e:
            raise SDKError(f"Failed to load pipeline from '{path}': {e}")

    def execute_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 300000
    ) -> Any:
        """
        Execute a pipeline with provided inputs.

        Configures the pipeline with the SDK's inference engine and
        executes it.

        Args:
            pipeline: Pipeline to execute.
            inputs: Input data dictionary.
            timeout_ms: Maximum execution time in milliseconds.

        Returns:
            PipelineResult with execution results.

        Example:
            sdk = SDK()
            pipeline = sdk.create_simple_pipeline("test", ["mobilenet_v2"])
            result = sdk.execute_pipeline(pipeline, {"input_0": data})
        """
        # Set up the executor to use our inference engine
        engine = self._get_inference_engine()

        def stage_executor(stage: "PipelineStage", stage_inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a stage using the inference engine."""
            result = engine.infer(
                model_name=stage.model,
                inputs=stage_inputs,
                timeout_ms=timeout_ms
            )
            return result.outputs

        pipeline.set_executor(stage_executor)
        return pipeline.execute(inputs, timeout_ms)

    def __repr__(self) -> str:
        return f"SDK(sdk_root={self.sdk_root!r})"

    # =========================================================================
    # Convenience Functions for Common ML Tasks
    # =========================================================================

    # Default models for convenience functions
    DEFAULT_CLASSIFICATION_MODEL = "mobilenet_v2"
    STYLE_TRANSFER_MODELS = ["la_muse", "udnie", "mirror", "wave_crop", "des_glaneuses"]

    def classify(
        self,
        image: Any,
        model: Optional[str] = None,
        top_k: int = 5,
        timeout_ms: int = 300000
    ) -> Dict[str, Any]:
        """
        Run image classification on an input image.

        A convenience function for running image classification using
        MobileNet V2 or another classification model.

        Args:
            image: Input image data. Can be:
                   - numpy array (HWC or CHW format)
                   - path to image file (str or Path)
                   - dict with "data" key containing image data
            model: Classification model name. Defaults to "mobilenet_v2".
            top_k: Number of top predictions to return.
            timeout_ms: Maximum execution time in milliseconds.

        Returns:
            Dict containing:
                - "predictions": List of (class_id, score) tuples
                - "top_class": Predicted class with highest score
                - "confidence": Confidence score for top class
                - "execution_time_ms": Time taken for inference
                - "model": Model name used

        Example:
            sdk = SDK()

            # Classify from file path
            result = sdk.classify("/path/to/image.jpg")

            # Classify from numpy array
            import numpy as np
            image = np.random.rand(224, 224, 3).astype(np.float32)
            result = sdk.classify(image)

            print(f"Top class: {result['top_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
        """
        # Use default model if not specified
        model_name = model or self.DEFAULT_CLASSIFICATION_MODEL

        # Validate model exists
        model_path = self.get_model_path(model_name)
        if model_path is None:
            raise SDKError(
                f"Classification model '{model_name}' not found. "
                f"Available models: {self.list_models()}"
            )

        # Prepare inputs
        inputs = self._prepare_image_input(image)

        # Run inference
        engine = self._get_inference_engine()
        result = engine.infer(
            model_name=model_name,
            inputs=inputs,
            timeout_ms=timeout_ms
        )

        # Format classification result
        return self._format_classification_result(result, model_name, top_k)

    def style_transfer(
        self,
        image: Any,
        style: str = "la_muse",
        output_size: Optional[tuple] = None,
        timeout_ms: int = 300000
    ) -> Dict[str, Any]:
        """
        Apply artistic style transfer to an input image.

        A convenience function for running neural style transfer using
        pre-trained style models like La Muse, Udnie, etc.

        Available styles:
            - "la_muse": Based on La Muse by Pablo Picasso
            - "udnie": Based on Udnie by Francis Picabia
            - "mirror": Mirror-like abstract style
            - "wave_crop": Based on The Great Wave (cropped)
            - "des_glaneuses": Based on Des Glaneuses by Jean-François Millet

        Args:
            image: Input image data. Can be:
                   - numpy array (HWC or CHW format)
                   - path to image file (str or Path)
                   - dict with "data" key containing image data
            style: Style model name. Defaults to "la_muse".
            output_size: Optional (width, height) tuple for output size.
                        If None, uses input image size.
            timeout_ms: Maximum execution time in milliseconds.

        Returns:
            Dict containing:
                - "stylized_image": Stylized image data (numpy array)
                - "style": Style model name used
                - "input_size": Original input image size
                - "output_size": Output image size
                - "execution_time_ms": Time taken for inference
                - "output_path": Path to output file (if saved)

        Example:
            sdk = SDK()

            # Apply La Muse style
            result = sdk.style_transfer("/path/to/photo.jpg", style="la_muse")

            # Apply Udnie style with custom output size
            result = sdk.style_transfer(
                image_array,
                style="udnie",
                output_size=(512, 512)
            )

            print(f"Style: {result['style']}")
            print(f"Time: {result['execution_time_ms']:.0f}ms")
        """
        # Validate style model exists
        model_path = self.get_model_path(style)
        if model_path is None:
            available_styles = self.list_available_styles()
            raise SDKError(
                f"Style model '{style}' not found. "
                f"Available styles: {available_styles or self.STYLE_TRANSFER_MODELS}"
            )

        # Prepare inputs
        inputs = self._prepare_image_input(image)
        if output_size:
            inputs["output_size"] = list(output_size)

        # Run inference
        engine = self._get_inference_engine()
        result = engine.infer(
            model_name=style,
            inputs=inputs,
            timeout_ms=timeout_ms
        )

        # Format style transfer result
        return self._format_style_transfer_result(result, style, image, output_size)

    def _prepare_image_input(self, image: Any) -> Dict[str, Any]:
        """
        Prepare image input for inference.

        Handles various input formats including file paths, numpy arrays,
        and dictionaries.

        Args:
            image: Input image in various formats.

        Returns:
            Dict with prepared input data.
        """
        inputs: Dict[str, Any] = {}

        if isinstance(image, dict):
            # Already a dict, pass through
            inputs.update(image)
        elif isinstance(image, (str, Path)):
            # File path
            image_path = Path(image)
            if not image_path.exists():
                raise SDKError(f"Image file not found: {image}")
            inputs["image_path"] = str(image_path)
        else:
            # Assume numpy array or array-like
            inputs["image"] = image

        return inputs

    def _format_classification_result(
        self,
        result: "InferenceResult",
        model_name: str,
        top_k: int
    ) -> Dict[str, Any]:
        """
        Format inference result as classification output.

        Args:
            result: Raw inference result.
            model_name: Name of the model used.
            top_k: Number of top predictions.

        Returns:
            Formatted classification result dict.
        """
        # Extract predictions from outputs
        outputs = result.outputs or {}

        # Handle various output formats
        predictions = []
        if "predictions" in outputs:
            predictions = outputs["predictions"]
        elif "logits" in outputs:
            logits = outputs["logits"]
            # Convert logits to predictions (would apply softmax in real impl)
            if hasattr(logits, 'tolist'):
                logits = logits.tolist()
            if isinstance(logits, list):
                predictions = [(i, float(score)) for i, score in enumerate(logits)]
                predictions.sort(key=lambda x: x[1], reverse=True)

        # Get top-k predictions
        top_predictions = predictions[:top_k] if predictions else []

        # Extract top class and confidence
        top_class = top_predictions[0][0] if top_predictions else -1
        confidence = top_predictions[0][1] if top_predictions else 0.0

        return {
            "predictions": top_predictions,
            "top_class": top_class,
            "confidence": confidence,
            "execution_time_ms": result.execution_time_ms,
            "model": model_name,
            "success": result.success,
            "metadata": result.metadata,
        }

    def _format_style_transfer_result(
        self,
        result: "InferenceResult",
        style: str,
        original_image: Any,
        output_size: Optional[tuple]
    ) -> Dict[str, Any]:
        """
        Format inference result as style transfer output.

        Args:
            result: Raw inference result.
            style: Style model name used.
            original_image: Original input image.
            output_size: Requested output size.

        Returns:
            Formatted style transfer result dict.
        """
        outputs = result.outputs or {}

        # Extract stylized image from outputs
        stylized_image = outputs.get("stylized_image", outputs.get("output", None))

        # Determine sizes
        input_size = None
        if hasattr(original_image, 'shape'):
            shape = original_image.shape
            if len(shape) >= 2:
                input_size = (shape[1], shape[0])  # (width, height)

        actual_output_size = output_size
        if stylized_image is not None and hasattr(stylized_image, 'shape'):
            shape = stylized_image.shape
            if len(shape) >= 2:
                actual_output_size = (shape[1], shape[0])

        # Get output path if available
        output_path = outputs.get("output_path", outputs.get("path", None))

        return {
            "stylized_image": stylized_image,
            "style": style,
            "input_size": input_size,
            "output_size": actual_output_size,
            "execution_time_ms": result.execution_time_ms,
            "output_path": output_path,
            "success": result.success,
            "metadata": result.metadata,
        }

    def list_available_styles(self) -> List[str]:
        """
        List available style transfer models.

        Returns:
            List of available style model names.
        """
        available = []
        for style in self.STYLE_TRANSFER_MODELS:
            if self.get_model_path(style) is not None:
                available.append(style)
        return available


# =============================================================================
# Module-level Convenience Functions
# =============================================================================

# Singleton SDK instance for module-level convenience functions
_default_sdk: Optional[SDK] = None


def _get_default_sdk() -> SDK:
    """Get or create the default SDK instance."""
    global _default_sdk
    if _default_sdk is None:
        _default_sdk = SDK()
    return _default_sdk


def classify(
    image: Any,
    model: Optional[str] = None,
    top_k: int = 5,
    timeout_ms: int = 300000,
    sdk: Optional[SDK] = None
) -> Dict[str, Any]:
    """
    Run image classification on an input image.

    A module-level convenience function for running image classification.
    Uses a default SDK instance if none is provided.

    Args:
        image: Input image data (numpy array, file path, or dict).
        model: Classification model name. Defaults to "mobilenet_v2".
        top_k: Number of top predictions to return.
        timeout_ms: Maximum execution time in milliseconds.
        sdk: Optional SDK instance to use. Creates default if not provided.

    Returns:
        Dict with classification results including predictions, top_class,
        confidence, and execution_time_ms.

    Example:
        import vulkan_ml_sdk as vms

        result = vms.classify("/path/to/image.jpg")
        print(f"Top class: {result['top_class']}")
        print(f"Confidence: {result['confidence']:.2%}")

        # With custom model
        result = vms.classify(image_array, model="fire_detection")
    """
    sdk_instance = sdk or _get_default_sdk()
    return sdk_instance.classify(
        image=image,
        model=model,
        top_k=top_k,
        timeout_ms=timeout_ms
    )


def style_transfer(
    image: Any,
    style: str = "la_muse",
    output_size: Optional[tuple] = None,
    timeout_ms: int = 300000,
    sdk: Optional[SDK] = None
) -> Dict[str, Any]:
    """
    Apply artistic style transfer to an input image.

    A module-level convenience function for running neural style transfer.
    Uses a default SDK instance if none is provided.

    Available styles:
        - "la_muse": Based on La Muse by Pablo Picasso
        - "udnie": Based on Udnie by Francis Picabia
        - "mirror": Mirror-like abstract style
        - "wave_crop": Based on The Great Wave (cropped)
        - "des_glaneuses": Based on Des Glaneuses by Jean-François Millet

    Args:
        image: Input image data (numpy array, file path, or dict).
        style: Style model name. Defaults to "la_muse".
        output_size: Optional (width, height) tuple for output size.
        timeout_ms: Maximum execution time in milliseconds.
        sdk: Optional SDK instance to use. Creates default if not provided.

    Returns:
        Dict with style transfer results including stylized_image, style,
        input_size, output_size, and execution_time_ms.

    Example:
        import vulkan_ml_sdk as vms

        result = vms.style_transfer("/path/to/photo.jpg", style="udnie")
        print(f"Style: {result['style']}")
        print(f"Time: {result['execution_time_ms']:.0f}ms")
    """
    sdk_instance = sdk or _get_default_sdk()
    return sdk_instance.style_transfer(
        image=image,
        style=style,
        output_size=output_size,
        timeout_ms=timeout_ms
    )
