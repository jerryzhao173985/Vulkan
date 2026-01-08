#!/usr/bin/env python3
"""
Inference module wrapping scenario-runner with sync and async execution.

This module provides a high-level Python interface for running ML inference
using the scenario-runner executable on Vulkan compute.

Features:
- Synchronous inference execution with timeout support
- Asynchronous inference with futures and callback patterns
- Model loading with automatic scenario generation
- Result parsing and structured output
- Performance profiling integration

Example usage:
    from vulkan_ml_sdk.inference import InferenceEngine, AsyncInference

    # Synchronous inference
    engine = InferenceEngine("builds/ARM-ML-SDK-Complete")
    if engine.is_available():
        result = engine.infer("mobilenet_v2", input_data)

    # Asynchronous inference
    async_engine = AsyncInference(engine)
    future = async_engine.submit("mobilenet_v2", input_data)
    future.add_callback(lambda r: print(f"Done: {r}"))
    result = future.result()
"""

import os
import json
import subprocess
import tempfile
import time
import threading
import uuid
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable


class InferenceError(Exception):
    """Base exception for inference-related errors."""
    pass


class ModelLoadError(InferenceError):
    """Error loading or initializing a model."""
    pass


class ExecutionError(InferenceError):
    """Error during inference execution."""
    pass


class InferenceTimeoutError(InferenceError):
    """Inference execution timed out."""
    pass


# Alias for backwards compatibility (but prefer InferenceTimeoutError)
TimeoutError = InferenceTimeoutError


class ValidationError(InferenceError):
    """Error during input validation."""
    pass


class InferenceResult:
    """
    Result of an inference execution.

    Attributes:
        success: Whether inference completed successfully.
        model_name: Name of the model used.
        outputs: Output data from inference.
        execution_time_ms: Execution time in milliseconds.
        metadata: Additional execution metadata.
    """

    def __init__(
        self,
        success: bool,
        model_name: str,
        outputs: Optional[Dict[str, Any]] = None,
        execution_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize InferenceResult.

        Args:
            success: Whether inference completed successfully.
            model_name: Name of the model used.
            outputs: Output data from inference.
            execution_time_ms: Execution time in milliseconds.
            metadata: Additional execution metadata.
        """
        self.success = success
        self.model_name = model_name
        self.outputs = outputs or {}
        self.execution_time_ms = execution_time_ms
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "model_name": self.model_name,
            "outputs": self.outputs,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"InferenceResult(success={self.success}, "
            f"model={self.model_name!r}, "
            f"time={self.execution_time_ms:.2f}ms)"
        )


class InferenceEngine:
    """
    Synchronous inference engine wrapping scenario-runner.

    Provides a high-level interface for running ML inference on Vulkan
    compute shaders using the scenario-runner executable.

    Example:
        engine = InferenceEngine("builds/ARM-ML-SDK-Complete")
        if engine.is_available():
            result = engine.infer("mobilenet_v2", input_data)
            if result.success:
                print(f"Inference completed in {result.execution_time_ms}ms")
    """

    # Default timeout for inference operations (5 minutes)
    DEFAULT_TIMEOUT_MS = 300000

    def __init__(self, sdk_root: Optional[Union[str, Path]] = None):
        """
        Initialize the InferenceEngine.

        Args:
            sdk_root: Path to the SDK root directory. If not provided,
                      auto-detects from package location.
        """
        self._sdk_root = self._resolve_sdk_root(sdk_root)
        self._env = self._setup_environment()
        self._loaded_models: Dict[str, Dict[str, Any]] = {}
        self._available: Optional[bool] = None

    def _resolve_sdk_root(self, sdk_root: Optional[Union[str, Path]]) -> Path:
        """
        Resolve the SDK root directory.

        Args:
            sdk_root: User-provided SDK root or None for auto-detection.

        Returns:
            Path to SDK root directory.
        """
        if sdk_root:
            return Path(sdk_root).resolve()

        # Auto-detect: lib/python/vulkan_ml_sdk/inference.py -> SDK_ROOT
        current_file = Path(__file__).resolve()
        return current_file.parent.parent.parent.parent

    def _setup_environment(self) -> Dict[str, str]:
        """
        Configure environment variables for Vulkan and SDK libraries.

        Returns:
            Environment dictionary for subprocess calls.
        """
        env = os.environ.copy()

        lib_path = str(self.lib_dir)
        system_lib = "/usr/local/lib"

        # Set DYLD_LIBRARY_PATH for macOS
        current_dyld = env.get("DYLD_LIBRARY_PATH", "")
        paths_to_add = [system_lib, lib_path]

        # Build new path, avoiding duplicates
        existing_paths = current_dyld.split(":") if current_dyld else []
        new_paths = []
        for path in paths_to_add + existing_paths:
            if path and path not in new_paths:
                new_paths.append(path)

        env["DYLD_LIBRARY_PATH"] = ":".join(new_paths)

        return env

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
    def scenario_runner_path(self) -> Path:
        """Path to the scenario-runner executable."""
        return self.bin_dir / "scenario-runner"

    def is_available(self) -> bool:
        """
        Check if the inference engine is available and ready.

        Verifies that:
        - SDK root directory exists
        - scenario-runner executable exists and is executable
        - Required directories are present

        Returns:
            True if engine is ready for inference, False otherwise.
        """
        if self._available is not None:
            return self._available

        self._available = self._check_availability()
        return self._available

    def _check_availability(self) -> bool:
        """
        Perform availability checks.

        Returns:
            True if all checks pass, False otherwise.
        """
        # Check SDK root exists
        if not self._sdk_root.exists():
            return False

        # Check scenario-runner exists
        runner_path = self.scenario_runner_path
        if not runner_path.exists():
            return False

        # Check executable permissions
        if not os.access(runner_path, os.X_OK):
            return False

        # Verify it can run (--version or --help)
        try:
            result = subprocess.run(
                [str(runner_path), "--version"],
                capture_output=True,
                text=True,
                env=self._env,
                timeout=10
            )
            # Accept both success and known error codes
            # (some versions may not have --version)
            return result.returncode in [0, 1]
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return False

    def get_version(self) -> Optional[str]:
        """
        Get the scenario-runner version.

        Returns:
            Version string or None if unable to retrieve.
        """
        if not self.is_available():
            return None

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
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
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

    def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        Load and prepare a model for inference.

        Args:
            model_name: Name of the model to load.

        Returns:
            Dict with model information.

        Raises:
            ValidationError: If model_name is empty or not a string.
            ModelLoadError: If model cannot be loaded.
        """
        # Validate input
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("model_name must be a non-empty string")

        model_name = model_name.strip()
        model_path = self.get_model_path(model_name)
        if model_path is None:
            raise ModelLoadError(f"Model not found: {model_name}")

        # Store model info
        model_info = {
            "name": model_name,
            "path": str(model_path),
            "size": model_path.stat().st_size,
            "loaded_at": time.time(),
        }

        self._loaded_models[model_name] = model_info
        return model_info

    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is loaded.

        Args:
            model_name: Name of the model.

        Returns:
            True if model is loaded, False otherwise.
        """
        return model_name in self._loaded_models

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from the engine.

        Args:
            model_name: Name of the model to unload.

        Returns:
            True if model was unloaded, False if not found.
        """
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            return True
        return False

    def run_scenario(
        self,
        scenario: Union[str, Path, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        dry_run: bool = False,
        profiling_dump_path: Optional[Union[str, Path]] = None,
        pipeline_caching: bool = False
    ) -> InferenceResult:
        """
        Run a scenario through the scenario-runner.

        Args:
            scenario: Path to scenario JSON file or scenario dict.
            output_dir: Directory for output files.
            timeout_ms: Maximum execution time in milliseconds (must be > 0).
            dry_run: If True, validate scenario without executing.
            profiling_dump_path: Path for performance metrics output.
            pipeline_caching: If True, enable shader pipeline caching.

        Returns:
            InferenceResult with execution results.

        Raises:
            ValidationError: If timeout_ms is not a positive integer.
            ExecutionError: If scenario execution fails critically.
            TimeoutError: If execution exceeds timeout.
        """
        # Validate inputs
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValidationError(f"timeout_ms must be a positive integer, got {timeout_ms}")

        if not self.is_available():
            raise ExecutionError("Inference engine is not available")

        start_time = time.time()

        # Handle scenario as dict - write to temp file
        temp_scenario_file = None
        if isinstance(scenario, dict):
            fd, temp_scenario_file = tempfile.mkstemp(suffix=".json")
            with os.fdopen(fd, 'w') as f:
                json.dump(scenario, f, indent=2)
            scenario_path = temp_scenario_file
        else:
            scenario_path = str(scenario)

        # Create temp output dir if not specified
        temp_output_dir = None
        if output_dir is None:
            temp_output_dir = tempfile.mkdtemp(prefix="vulkan_ml_")
            output_dir = temp_output_dir

        try:
            # Build command
            cmd = [str(self.scenario_runner_path), "--scenario", scenario_path]

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
            timeout_seconds = timeout_ms / 1000.0
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._env,
                timeout=timeout_seconds
            )

            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000

            # Parse outputs
            outputs = self._parse_outputs(output_path)

            # Build metadata
            metadata = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "dry_run": dry_run,
            }

            return InferenceResult(
                success=(result.returncode == 0),
                model_name=self._extract_model_name(scenario),
                outputs=outputs,
                execution_time_ms=execution_time_ms,
                metadata=metadata
            )

        except subprocess.TimeoutExpired as e:
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            raise InferenceTimeoutError(
                f"Inference timed out after {execution_time_ms:.0f}ms "
                f"(limit: {timeout_ms}ms)"
            )
        except subprocess.SubprocessError as e:
            raise ExecutionError(f"Failed to execute scenario-runner: {e}")
        finally:
            # Clean up temp files
            if temp_scenario_file and os.path.exists(temp_scenario_file):
                os.unlink(temp_scenario_file)
            # Note: temp_output_dir is left for the caller to handle if needed

    def _extract_model_name(
        self,
        scenario: Union[str, Path, Dict[str, Any]]
    ) -> str:
        """Extract model name from scenario."""
        if isinstance(scenario, dict):
            return scenario.get("name", scenario.get("model", "unknown"))
        return Path(scenario).stem

    def _parse_outputs(self, output_dir: Path) -> Dict[str, Any]:
        """
        Parse outputs from the output directory.

        Args:
            output_dir: Path to output directory.

        Returns:
            Dict with parsed outputs.
        """
        outputs = {}

        if not output_dir.exists():
            return outputs

        # Look for common output files
        for output_file in output_dir.iterdir():
            if output_file.is_file():
                name = output_file.stem
                suffix = output_file.suffix.lower()

                if suffix == ".json":
                    try:
                        with open(output_file, 'r') as f:
                            outputs[name] = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        outputs[name] = {"path": str(output_file)}
                elif suffix in (".npy", ".bin", ".dat"):
                    outputs[name] = {"path": str(output_file)}
                else:
                    outputs[name] = {"path": str(output_file)}

        return outputs

    def infer(
        self,
        model_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        profiling: bool = False
    ) -> InferenceResult:
        """
        Run inference on a model.

        This is a convenience method that generates a scenario and runs it.

        Args:
            model_name: Name of the model to use.
            inputs: Input data dictionary.
            timeout_ms: Maximum execution time in milliseconds (must be > 0).
            profiling: If True, enable performance profiling.

        Returns:
            InferenceResult with inference results.

        Raises:
            ValidationError: If model_name is empty or timeout_ms is invalid.
            ModelLoadError: If model is not found.
            ExecutionError: If inference fails.
        """
        # Validate inputs
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("model_name must be a non-empty string")
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValidationError(f"timeout_ms must be a positive integer, got {timeout_ms}")

        model_name = model_name.strip()
        model_path = self.get_model_path(model_name)
        if model_path is None:
            raise ModelLoadError(f"Model not found: {model_name}")

        # Generate scenario for the model
        scenario = self._generate_scenario(model_name, model_path, inputs)

        # Run inference
        profiling_path = None
        if profiling:
            fd, profiling_path = tempfile.mkstemp(suffix=".json")
            os.close(fd)

        try:
            result = self.run_scenario(
                scenario=scenario,
                timeout_ms=timeout_ms,
                profiling_dump_path=profiling_path
            )

            # Add profiling data if available
            if profiling_path and os.path.exists(profiling_path):
                try:
                    with open(profiling_path, 'r') as f:
                        result.metadata["profiling"] = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            return result

        finally:
            if profiling_path and os.path.exists(profiling_path):
                os.unlink(profiling_path)

    def _generate_scenario(
        self,
        model_name: str,
        model_path: Path,
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a scenario dictionary for a model.

        Args:
            model_name: Name of the model.
            model_path: Path to the model file.
            inputs: Optional input data.

        Returns:
            Scenario dictionary.
        """
        scenario = {
            "name": f"{model_name}_inference",
            "description": f"Inference scenario for {model_name}",
            "version": "1.0",
            "model": {
                "path": str(model_path),
                "format": "tflite"
            }
        }

        if inputs:
            scenario["inputs"] = inputs

        return scenario

    def validate_scenario(
        self,
        scenario: Union[str, Path, Dict[str, Any]]
    ) -> InferenceResult:
        """
        Validate a scenario without executing it.

        Args:
            scenario: Path to scenario JSON file or scenario dict.

        Returns:
            InferenceResult with validation results.
        """
        return self.run_scenario(scenario, dry_run=True)

    def get_info(self) -> Dict[str, Any]:
        """
        Get engine information summary.

        Returns:
            Dict with engine status, paths, and available resources.
        """
        return {
            "available": self.is_available(),
            "sdk_root": str(self.sdk_root),
            "version": self.get_version(),
            "paths": {
                "bin": str(self.bin_dir),
                "lib": str(self.lib_dir),
                "models": str(self.models_dir),
                "shaders": str(self.shaders_dir),
                "scenario_runner": str(self.scenario_runner_path),
            },
            "resources": {
                "models": self.list_models(),
                "shaders": self.list_shaders(),
                "model_count": len(self.list_models()),
                "shader_count": len(self.list_shaders()),
            },
            "loaded_models": list(self._loaded_models.keys()),
            "environment": {
                "DYLD_LIBRARY_PATH": self._env.get("DYLD_LIBRARY_PATH", ""),
            }
        }

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        return f"InferenceEngine(sdk_root={self.sdk_root!r}, status={status})"


class InferenceFuture:
    """
    A future representing an asynchronous inference operation.

    Provides a way to wait for inference results and attach callbacks
    that fire when inference completes.

    Attributes:
        task_id: Unique identifier for this inference task.
        model_name: Name of the model being used.
        submitted_at: Timestamp when inference was submitted.
    """

    def __init__(
        self,
        task_id: str,
        model_name: str,
        future: Future,
    ):
        """
        Initialize InferenceFuture.

        Args:
            task_id: Unique identifier for this task.
            model_name: Name of the model.
            future: Underlying concurrent.futures.Future object.
        """
        self.task_id = task_id
        self.model_name = model_name
        self.submitted_at = time.time()
        self._future = future
        self._callbacks: List[Callable[[InferenceResult], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        self._lock = threading.Lock()
        self._result: Optional[InferenceResult] = None
        self._exception: Optional[Exception] = None

        # Attach internal callback to handle completion
        self._future.add_done_callback(self._on_complete)

    def _on_complete(self, future: Future) -> None:
        """Handle future completion and invoke callbacks."""
        with self._lock:
            try:
                self._result = future.result()
                self._exception = None
                # Invoke success callbacks
                for callback in self._callbacks:
                    try:
                        callback(self._result)
                    except Exception:
                        pass  # Don't let callback errors propagate
            except Exception as e:
                self._exception = e
                self._result = None
                # Invoke error callbacks
                for error_callback in self._error_callbacks:
                    try:
                        error_callback(e)
                    except Exception:
                        pass  # Don't let callback errors propagate

    def add_callback(
        self,
        callback: Callable[[InferenceResult], None]
    ) -> "InferenceFuture":
        """
        Add a callback to be invoked when inference succeeds.

        If inference has already completed successfully, the callback
        is invoked immediately.

        Args:
            callback: Function to call with the InferenceResult.

        Returns:
            Self for method chaining.
        """
        with self._lock:
            if self._result is not None:
                # Already completed successfully, invoke immediately
                try:
                    callback(self._result)
                except Exception:
                    pass
            elif self._exception is None:
                # Not yet complete, add to pending callbacks
                self._callbacks.append(callback)
        return self

    def add_error_callback(
        self,
        callback: Callable[[Exception], None]
    ) -> "InferenceFuture":
        """
        Add a callback to be invoked when inference fails.

        If inference has already failed, the callback is invoked immediately.

        Args:
            callback: Function to call with the exception.

        Returns:
            Self for method chaining.
        """
        with self._lock:
            if self._exception is not None:
                # Already failed, invoke immediately
                try:
                    callback(self._exception)
                except Exception:
                    pass
            elif self._result is None:
                # Not yet complete, add to pending callbacks
                self._error_callbacks.append(callback)
        return self

    def result(self, timeout: Optional[float] = None) -> InferenceResult:
        """
        Wait for and return the inference result.

        Args:
            timeout: Maximum time to wait in seconds. None for no limit.
                     Must be a non-negative number if provided.

        Returns:
            InferenceResult from the inference operation.

        Raises:
            ValidationError: If timeout is negative.
            InferenceTimeoutError: If timeout is exceeded.
            InferenceError: If inference failed.
        """
        # Validate timeout
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout < 0):
            raise ValidationError(f"timeout must be a non-negative number, got {timeout}")

        try:
            return self._future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Inference timed out after {timeout}s")
        except concurrent.futures.CancelledError:
            raise ExecutionError("Inference was cancelled")
        except Exception as e:
            # Re-raise if it's already an InferenceError subclass
            if isinstance(e, InferenceError):
                raise
            raise ExecutionError(f"Inference failed: {e}")

    def done(self) -> bool:
        """
        Check if inference has completed.

        Returns:
            True if inference is complete (success or failure).
        """
        return self._future.done()

    def running(self) -> bool:
        """
        Check if inference is currently running.

        Returns:
            True if inference is in progress.
        """
        return self._future.running()

    def cancelled(self) -> bool:
        """
        Check if inference was cancelled.

        Returns:
            True if inference was cancelled.
        """
        return self._future.cancelled()

    def cancel(self) -> bool:
        """
        Attempt to cancel the inference operation.

        Returns:
            True if cancellation was successful.
        """
        return self._future.cancel()

    def wait_time_ms(self) -> float:
        """
        Get the elapsed time since submission.

        Returns:
            Time in milliseconds since submit was called.
        """
        return (time.time() - self.submitted_at) * 1000

    def __repr__(self) -> str:
        status = "done" if self.done() else "pending"
        return (
            f"InferenceFuture(task_id={self.task_id!r}, "
            f"model={self.model_name!r}, status={status})"
        )


class AsyncInference:
    """
    Asynchronous inference engine using futures and callbacks.

    Wraps InferenceEngine to provide non-blocking inference with
    futures that support callbacks for completion notification.

    Example:
        engine = InferenceEngine("builds/ARM-ML-SDK-Complete")
        async_engine = AsyncInference(engine, max_workers=4)

        # Submit inference and get a future
        future = async_engine.submit("mobilenet_v2", input_data)

        # Add callbacks for success and error
        future.add_callback(lambda r: print(f"Success: {r}"))
        future.add_error_callback(lambda e: print(f"Error: {e}"))

        # Or wait for result
        result = future.result(timeout=30)

        # Batch submission
        futures = async_engine.submit_batch(
            [("mobilenet_v2", data1), ("fire_detection", data2)]
        )

        # Cleanup when done
        async_engine.shutdown()
    """

    # Default number of worker threads
    DEFAULT_MAX_WORKERS = 4

    def __init__(
        self,
        engine: Optional[InferenceEngine] = None,
        sdk_root: Optional[Union[str, Path]] = None,
        max_workers: int = DEFAULT_MAX_WORKERS
    ):
        """
        Initialize AsyncInference.

        Args:
            engine: Existing InferenceEngine instance to use.
            sdk_root: SDK root path (used if engine not provided).
            max_workers: Maximum number of concurrent inference threads (must be > 0).

        Raises:
            ValidationError: If max_workers is not a positive integer.
        """
        # Validate max_workers
        if not isinstance(max_workers, int) or max_workers <= 0:
            raise ValidationError(f"max_workers must be a positive integer, got {max_workers}")

        if engine is not None:
            self._engine = engine
        else:
            self._engine = InferenceEngine(sdk_root)

        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="vulkan_ml_async"
        )
        self._pending_tasks: Dict[str, InferenceFuture] = {}
        self._lock = threading.Lock()
        self._shutdown = False

    @property
    def engine(self) -> InferenceEngine:
        """Get the underlying InferenceEngine."""
        return self._engine

    @property
    def max_workers(self) -> int:
        """Maximum number of concurrent inference threads."""
        return self._max_workers

    def is_available(self) -> bool:
        """Check if async inference is available."""
        return self._engine.is_available() and not self._shutdown

    def submit(
        self,
        model_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        timeout_ms: int = InferenceEngine.DEFAULT_TIMEOUT_MS,
        profiling: bool = False,
        callback: Optional[Callable[[InferenceResult], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None
    ) -> InferenceFuture:
        """
        Submit an inference request for asynchronous execution.

        Args:
            model_name: Name of the model to use.
            inputs: Input data dictionary.
            timeout_ms: Maximum execution time in milliseconds (must be > 0).
            profiling: If True, enable performance profiling.
            callback: Optional callback for successful completion.
            error_callback: Optional callback for errors.

        Returns:
            InferenceFuture for tracking the inference operation.

        Raises:
            ValidationError: If model_name is empty or timeout_ms is invalid.
            RuntimeError: If engine is shut down.
            ModelLoadError: If model is not found.
        """
        # Validate inputs
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("model_name must be a non-empty string")
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValidationError(f"timeout_ms must be a positive integer, got {timeout_ms}")

        model_name = model_name.strip()

        if self._shutdown:
            raise RuntimeError("AsyncInference has been shut down")

        # Validate model exists before submitting
        if self._engine.get_model_path(model_name) is None:
            raise ModelLoadError(f"Model not found: {model_name}")

        # Generate unique task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Submit to thread pool
        future = self._executor.submit(
            self._run_inference,
            model_name,
            inputs,
            timeout_ms,
            profiling
        )

        # Create InferenceFuture wrapper
        inference_future = InferenceFuture(task_id, model_name, future)

        # Add callbacks if provided
        if callback:
            inference_future.add_callback(callback)
        if error_callback:
            inference_future.add_error_callback(error_callback)

        # Track the pending task
        with self._lock:
            self._pending_tasks[task_id] = inference_future

        # Clean up completed tasks on completion
        inference_future.add_callback(
            lambda _: self._cleanup_task(task_id)
        )
        inference_future.add_error_callback(
            lambda _: self._cleanup_task(task_id)
        )

        return inference_future

    def submit_batch(
        self,
        requests: List[tuple],
        timeout_ms: int = InferenceEngine.DEFAULT_TIMEOUT_MS,
        profiling: bool = False,
        callback: Optional[Callable[[InferenceResult], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None
    ) -> List[InferenceFuture]:
        """
        Submit multiple inference requests as a batch.

        Args:
            requests: List of (model_name, inputs) tuples.
            timeout_ms: Maximum execution time per inference (must be > 0).
            profiling: If True, enable performance profiling.
            callback: Optional callback for each successful completion.
            error_callback: Optional callback for each error.

        Returns:
            List of InferenceFutures for tracking each operation.

        Raises:
            ValidationError: If requests is not a list or timeout_ms is invalid.
        """
        # Validate inputs
        if not isinstance(requests, list):
            raise ValidationError("requests must be a list")
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValidationError(f"timeout_ms must be a positive integer, got {timeout_ms}")

        futures = []
        for request in requests:
            if isinstance(request, tuple) and len(request) >= 1:
                model_name = request[0]
                inputs = request[1] if len(request) > 1 else None
            else:
                model_name = str(request)
                inputs = None

            future = self.submit(
                model_name=model_name,
                inputs=inputs,
                timeout_ms=timeout_ms,
                profiling=profiling,
                callback=callback,
                error_callback=error_callback
            )
            futures.append(future)
        return futures

    def submit_scenario(
        self,
        scenario: Union[str, Path, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
        timeout_ms: int = InferenceEngine.DEFAULT_TIMEOUT_MS,
        callback: Optional[Callable[[InferenceResult], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None
    ) -> InferenceFuture:
        """
        Submit a scenario for asynchronous execution.

        Args:
            scenario: Path to scenario JSON or scenario dict.
            output_dir: Directory for output files.
            timeout_ms: Maximum execution time in milliseconds (must be > 0).
            callback: Optional callback for successful completion.
            error_callback: Optional callback for errors.

        Returns:
            InferenceFuture for tracking the scenario execution.

        Raises:
            ValidationError: If timeout_ms is not a positive integer.
            RuntimeError: If engine is shut down.
        """
        # Validate inputs
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValidationError(f"timeout_ms must be a positive integer, got {timeout_ms}")

        if self._shutdown:
            raise RuntimeError("AsyncInference has been shut down")

        task_id = f"scenario_{uuid.uuid4().hex[:8]}"

        # Extract model name from scenario
        if isinstance(scenario, dict):
            model_name = scenario.get("name", "scenario")
        else:
            model_name = Path(scenario).stem

        # Submit to thread pool
        future = self._executor.submit(
            self._run_scenario,
            scenario,
            output_dir,
            timeout_ms
        )

        # Create InferenceFuture wrapper
        inference_future = InferenceFuture(task_id, model_name, future)

        if callback:
            inference_future.add_callback(callback)
        if error_callback:
            inference_future.add_error_callback(error_callback)

        with self._lock:
            self._pending_tasks[task_id] = inference_future

        inference_future.add_callback(
            lambda _: self._cleanup_task(task_id)
        )
        inference_future.add_error_callback(
            lambda _: self._cleanup_task(task_id)
        )

        return inference_future

    def _run_inference(
        self,
        model_name: str,
        inputs: Optional[Dict[str, Any]],
        timeout_ms: int,
        profiling: bool
    ) -> InferenceResult:
        """Execute inference on a worker thread."""
        return self._engine.infer(
            model_name=model_name,
            inputs=inputs,
            timeout_ms=timeout_ms,
            profiling=profiling
        )

    def _run_scenario(
        self,
        scenario: Union[str, Path, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]],
        timeout_ms: int
    ) -> InferenceResult:
        """Execute scenario on a worker thread."""
        return self._engine.run_scenario(
            scenario=scenario,
            output_dir=output_dir,
            timeout_ms=timeout_ms
        )

    def _cleanup_task(self, task_id: str) -> None:
        """Remove completed task from pending tasks."""
        with self._lock:
            self._pending_tasks.pop(task_id, None)

    def get_pending_tasks(self) -> List[InferenceFuture]:
        """
        Get list of pending inference tasks.

        Returns:
            List of InferenceFuture objects for pending tasks.
        """
        with self._lock:
            return list(self._pending_tasks.values())

    def get_task(self, task_id: str) -> Optional[InferenceFuture]:
        """
        Get a specific pending task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            InferenceFuture if found, None otherwise.
        """
        with self._lock:
            return self._pending_tasks.get(task_id)

    def wait_all(
        self,
        futures: Optional[List[InferenceFuture]] = None,
        timeout: Optional[float] = None
    ) -> List[InferenceResult]:
        """
        Wait for multiple inference operations to complete.

        Args:
            futures: List of futures to wait for. If None, waits for
                     all pending tasks.
            timeout: Maximum time to wait in seconds. Must be non-negative
                     if provided.

        Returns:
            List of InferenceResults in the same order as futures.

        Raises:
            ValidationError: If timeout is negative.
            InferenceTimeoutError: If timeout is exceeded.
        """
        # Validate timeout
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout < 0):
            raise ValidationError(f"timeout must be a non-negative number, got {timeout}")

        if futures is None:
            with self._lock:
                futures = list(self._pending_tasks.values())

        results = []
        start_time = time.time()

        for future in futures:
            remaining = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = max(0, timeout - elapsed)
                if remaining <= 0:
                    raise InferenceTimeoutError("wait_all timed out")

            results.append(future.result(timeout=remaining))

        return results

    def cancel_all(self) -> int:
        """
        Attempt to cancel all pending tasks.

        Returns:
            Number of tasks successfully cancelled.
        """
        cancelled = 0
        with self._lock:
            for future in self._pending_tasks.values():
                if future.cancel():
                    cancelled += 1
        return cancelled

    def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        """
        Shutdown the async inference engine.

        Args:
            wait: If True, wait for pending tasks to complete.
            cancel_pending: If True, attempt to cancel pending tasks.
        """
        self._shutdown = True

        if cancel_pending:
            self.cancel_all()

        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "AsyncInference":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - shutdown executor."""
        self.shutdown(wait=True)

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        pending = len(self._pending_tasks)
        return (
            f"AsyncInference(status={status}, "
            f"workers={self._max_workers}, pending={pending})"
        )
