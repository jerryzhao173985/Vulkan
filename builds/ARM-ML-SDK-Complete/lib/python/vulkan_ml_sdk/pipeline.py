#!/usr/bin/env python3
"""
Pipeline orchestration for multi-model inference with data flow management.

This module provides pipeline capabilities for chaining multiple ML models
together with automatic data flow between stages.

Features:
- Pipeline stage management with model chaining
- Data flow connections between model outputs and inputs
- Parallel and sequential execution strategies
- Pipeline validation and optimization
- Resource management for Vulkan compute

Example usage:
    from vulkan_ml_sdk.pipeline import Pipeline, PipelineStage

    # Create a pipeline
    pipeline = Pipeline(name="image_classifier")

    # Add stages
    pipeline.add_stage(PipelineStage(
        name="preprocess",
        model="preprocessor",
        inputs=["raw_image"],
        outputs=["normalized_image"]
    ))

    pipeline.add_stage(PipelineStage(
        name="classify",
        model="mobilenet_v2",
        inputs=["normalized_image"],
        outputs=["predictions"]
    ))

    # Validate and execute
    pipeline.validate()
    result = pipeline.execute({"raw_image": image_data})
"""

import json
import time
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict


class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    pass


class ValidationError(PipelineError):
    """Error during input validation."""
    pass


class StageNotFoundError(PipelineError):
    """Raised when a pipeline stage cannot be found."""
    pass


class DataFlowError(PipelineError):
    """Raised when data flow between stages is invalid."""
    pass


class PipelineValidationError(PipelineError):
    """Raised when pipeline validation fails."""
    pass


class PipelineExecutionError(PipelineError):
    """Raised when pipeline execution fails."""
    pass


@dataclass
class OptimizationConfig:
    """
    Configuration for pipeline optimization.

    Attributes:
        enable_quantization: Enable quantization optimization.
        quantization_bits: Bit width for quantization (8 or 16).
        enable_fusion: Enable operator fusion optimization.
        fusion_patterns: List of fusion patterns to apply.
        use_fp16: Use FP16 acceleration where supported.
        use_simdgroup_operations: Use SIMD group operations (Apple Silicon).
        tile_size: Tile size for tiled operations.
        threadgroup_memory: Threadgroup memory size in bytes.
    """
    enable_quantization: bool = False
    quantization_bits: int = 8
    enable_fusion: bool = False
    fusion_patterns: List[str] = field(default_factory=lambda: [
        "conv_bn_relu",      # Conv2D + BatchNorm + ReLU
        "conv_relu",         # Conv2D + ReLU
        "matmul_add_relu",   # MatMul + Add + ReLU
        "linear_relu",       # Linear + ReLU
    ])
    use_fp16: bool = True
    use_simdgroup_operations: bool = True
    tile_size: int = 32
    threadgroup_memory: int = 32768

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationConfig":
        """Create config from dictionary, ignoring extra keys."""
        valid_fields = {
            "enable_quantization", "quantization_bits", "enable_fusion",
            "fusion_patterns", "use_fp16", "use_simdgroup_operations",
            "tile_size", "threadgroup_memory"
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class OptimizationResult:
    """
    Result from pipeline optimization.

    Attributes:
        success: Whether optimization was successful.
        optimizations_applied: List of optimizations that were applied.
        quantized_stages: List of stage names that were quantized.
        fused_operations: List of fused operation descriptions.
        estimated_speedup: Estimated speedup factor.
        memory_reduction: Estimated memory reduction factor.
        warnings: Any warnings from optimization.
    """
    success: bool
    optimizations_applied: List[str] = field(default_factory=list)
    quantized_stages: List[str] = field(default_factory=list)
    fused_operations: List[str] = field(default_factory=list)
    estimated_speedup: float = 1.0
    memory_reduction: float = 1.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


@dataclass
class PipelineStage:
    """
    Represents a single stage in an ML inference pipeline.

    A stage encapsulates a model execution with defined inputs and outputs,
    allowing for data flow connections to other stages.

    Attributes:
        name: Unique identifier for the stage.
        model: Name or path of the model to execute.
        inputs: List of input tensor/data names.
        outputs: List of output tensor/data names.
        config: Additional configuration for the stage.
        enabled: Whether the stage is enabled for execution.
    """
    name: str
    model: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert stage to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStage":
        """Create stage from dictionary, ignoring extra keys."""
        valid_fields = {"name", "model", "inputs", "outputs", "config", "enabled"}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def __repr__(self) -> str:
        return (
            f"PipelineStage(name={self.name!r}, model={self.model!r}, "
            f"inputs={self.inputs!r}, outputs={self.outputs!r})"
        )


@dataclass
class DataConnection:
    """
    Defines a data flow connection between pipeline stages.

    Attributes:
        source_stage: Name of the source stage.
        source_output: Output name from the source stage.
        target_stage: Name of the target stage.
        target_input: Input name for the target stage.
        transform: Optional transformation to apply during data flow.
    """
    source_stage: str
    source_output: str
    target_stage: str
    target_input: str
    transform: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert connection to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConnection":
        """Create connection from dictionary, ignoring extra keys."""
        valid_fields = {"source_stage", "source_output", "target_stage", "target_input", "transform"}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def __repr__(self) -> str:
        return (
            f"DataConnection({self.source_stage}.{self.source_output} -> "
            f"{self.target_stage}.{self.target_input})"
        )


@dataclass
class StageResult:
    """
    Result from executing a pipeline stage.

    Attributes:
        stage_name: Name of the stage that was executed.
        success: Whether the stage executed successfully.
        outputs: Output data from the stage.
        execution_time_ms: Time taken to execute in milliseconds.
        error: Error message if execution failed.
    """
    stage_name: str
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


@dataclass
class PipelineResult:
    """
    Result from executing a complete pipeline.

    Attributes:
        pipeline_name: Name of the executed pipeline.
        success: Whether all stages executed successfully.
        stage_results: Results from each stage.
        final_outputs: Final output data from the pipeline.
        total_time_ms: Total execution time in milliseconds.
        metadata: Additional execution metadata.
    """
    pipeline_name: str
    success: bool
    stage_results: List[StageResult] = field(default_factory=list)
    final_outputs: Dict[str, Any] = field(default_factory=dict)
    total_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = asdict(self)
        result["stage_results"] = [sr.to_dict() if hasattr(sr, 'to_dict') else sr
                                   for sr in self.stage_results]
        return result


class Pipeline:
    """
    Multi-model inference pipeline with data flow management.

    Provides orchestration for chaining multiple ML models together,
    with automatic data flow between stages and support for both
    sequential and parallel execution strategies.

    Example:
        pipeline = Pipeline(name="style_transfer")

        # Add preprocessing stage
        pipeline.add_stage(PipelineStage(
            name="normalize",
            model="normalizer",
            inputs=["image"],
            outputs=["normalized"]
        ))

        # Add main model stage
        pipeline.add_stage(PipelineStage(
            name="transfer",
            model="la_muse",
            inputs=["normalized"],
            outputs=["stylized"]
        ))

        # Connect stages
        pipeline.connect("normalize", "normalized", "transfer", "normalized")

        # Execute pipeline
        result = pipeline.execute({"image": input_image})
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: str = "",
        execution_strategy: str = "sequential"
    ):
        """
        Initialize a new Pipeline.

        Args:
            name: Unique name for the pipeline. Auto-generated if not provided.
            description: Human-readable description of the pipeline.
            execution_strategy: Strategy for stage execution ("sequential" or "parallel").
        """
        self._name = name or f"pipeline_{uuid.uuid4().hex[:8]}"
        self._description = description
        self._execution_strategy = execution_strategy
        self._stages: List[PipelineStage] = []
        self._connections: List[DataConnection] = []
        self._stage_map: Dict[str, PipelineStage] = {}
        self._lock = threading.RLock()
        self._created_at = time.time()
        self._executor: Optional[Callable] = None
        self._optimization_config: Optional[OptimizationConfig] = None
        self._optimization_result: Optional[OptimizationResult] = None

    @property
    def name(self) -> str:
        """Pipeline name."""
        return self._name

    @property
    def description(self) -> str:
        """Pipeline description."""
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        """Set pipeline description."""
        self._description = value

    @property
    def stages(self) -> List[PipelineStage]:
        """List of pipeline stages."""
        with self._lock:
            return list(self._stages)

    @property
    def connections(self) -> List[DataConnection]:
        """List of data connections between stages."""
        with self._lock:
            return list(self._connections)

    @property
    def execution_strategy(self) -> str:
        """Current execution strategy."""
        return self._execution_strategy

    @execution_strategy.setter
    def execution_strategy(self, value: str) -> None:
        """Set execution strategy."""
        if value not in ("sequential", "parallel"):
            raise ValueError("Execution strategy must be 'sequential' or 'parallel'")
        self._execution_strategy = value

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """
        Add a stage to the pipeline.

        Args:
            stage: PipelineStage to add.

        Returns:
            Self for method chaining.

        Raises:
            ValidationError: If stage is not a PipelineStage instance.
            PipelineError: If stage name already exists.
        """
        if not isinstance(stage, PipelineStage):
            raise ValidationError("stage must be a PipelineStage instance")

        with self._lock:
            if stage.name in self._stage_map:
                raise PipelineError(f"Stage already exists: {stage.name}")

            self._stages.append(stage)
            self._stage_map[stage.name] = stage

        return self

    def remove_stage(self, stage_name: str) -> bool:
        """
        Remove a stage from the pipeline.

        Args:
            stage_name: Name of the stage to remove.

        Returns:
            True if stage was removed, False if not found.

        Raises:
            ValidationError: If stage_name is empty or not a string.
        """
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValidationError("stage_name must be a non-empty string")

        with self._lock:
            if stage_name not in self._stage_map:
                return False

            stage = self._stage_map[stage_name]
            self._stages.remove(stage)
            del self._stage_map[stage_name]

            # Remove connections involving this stage
            self._connections = [
                c for c in self._connections
                if c.source_stage != stage_name and c.target_stage != stage_name
            ]

            return True

    def get_stage(self, stage_name: str) -> PipelineStage:
        """
        Get a stage by name.

        Args:
            stage_name: Name of the stage.

        Returns:
            PipelineStage object.

        Raises:
            ValidationError: If stage_name is empty or not a string.
            StageNotFoundError: If stage is not found.
        """
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValidationError("stage_name must be a non-empty string")

        with self._lock:
            if stage_name not in self._stage_map:
                raise StageNotFoundError(f"Stage not found: {stage_name}")
            return self._stage_map[stage_name]

    def has_stage(self, stage_name: str) -> bool:
        """
        Check if a stage exists.

        Args:
            stage_name: Name of the stage.

        Returns:
            True if stage exists, False otherwise.

        Raises:
            ValidationError: If stage_name is empty or not a string.
        """
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValidationError("stage_name must be a non-empty string")

        with self._lock:
            return stage_name in self._stage_map

    def connect(
        self,
        source_stage: str,
        source_output: str,
        target_stage: str,
        target_input: str,
        transform: Optional[str] = None
    ) -> "Pipeline":
        """
        Create a data flow connection between stages.

        Args:
            source_stage: Name of the source stage.
            source_output: Output name from the source stage.
            target_stage: Name of the target stage.
            target_input: Input name for the target stage.
            transform: Optional transformation to apply.

        Returns:
            Self for method chaining.

        Raises:
            ValidationError: If any required string parameter is empty.
            StageNotFoundError: If source or target stage doesn't exist.
            DataFlowError: If connection is invalid.
        """
        # Validate string parameters
        if not isinstance(source_stage, str) or not source_stage.strip():
            raise ValidationError("source_stage must be a non-empty string")
        if not isinstance(source_output, str) or not source_output.strip():
            raise ValidationError("source_output must be a non-empty string")
        if not isinstance(target_stage, str) or not target_stage.strip():
            raise ValidationError("target_stage must be a non-empty string")
        if not isinstance(target_input, str) or not target_input.strip():
            raise ValidationError("target_input must be a non-empty string")

        with self._lock:
            # Validate stages exist
            if source_stage not in self._stage_map:
                raise StageNotFoundError(f"Source stage not found: {source_stage}")
            if target_stage not in self._stage_map:
                raise StageNotFoundError(f"Target stage not found: {target_stage}")

            # Validate output exists in source stage
            src_stage = self._stage_map[source_stage]
            if source_output not in src_stage.outputs:
                raise DataFlowError(
                    f"Output '{source_output}' not found in stage '{source_stage}'"
                )

            # Validate input exists in target stage
            tgt_stage = self._stage_map[target_stage]
            if target_input not in tgt_stage.inputs:
                raise DataFlowError(
                    f"Input '{target_input}' not found in stage '{target_stage}'"
                )

            # Check for duplicate connections
            for conn in self._connections:
                if (conn.target_stage == target_stage and
                    conn.target_input == target_input):
                    raise DataFlowError(
                        f"Input '{target_input}' in stage '{target_stage}' "
                        "is already connected"
                    )

            # Create connection
            connection = DataConnection(
                source_stage=source_stage,
                source_output=source_output,
                target_stage=target_stage,
                target_input=target_input,
                transform=transform
            )
            self._connections.append(connection)

        return self

    def disconnect(
        self,
        source_stage: str,
        source_output: str,
        target_stage: str,
        target_input: str
    ) -> bool:
        """
        Remove a data flow connection.

        Args:
            source_stage: Name of the source stage.
            source_output: Output name from the source stage.
            target_stage: Name of the target stage.
            target_input: Input name for the target stage.

        Returns:
            True if connection was removed, False if not found.
        """
        with self._lock:
            for i, conn in enumerate(self._connections):
                if (conn.source_stage == source_stage and
                    conn.source_output == source_output and
                    conn.target_stage == target_stage and
                    conn.target_input == target_input):
                    del self._connections[i]
                    return True
            return False

    def validate(self) -> Dict[str, Any]:
        """
        Validate the pipeline structure and data flow.

        Checks for:
        - All required inputs are connected or provided
        - No circular dependencies
        - All stages have valid models

        Returns:
            Dict with validation results.

        Raises:
            PipelineValidationError: If validation fails with errors.
        """
        with self._lock:
            results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "stage_count": len(self._stages),
                "connection_count": len(self._connections),
            }

            if len(self._stages) == 0:
                results["warnings"].append("Pipeline has no stages")

            # Check for duplicate stage names (should be prevented by add_stage)
            stage_names = [s.name for s in self._stages]
            if len(stage_names) != len(set(stage_names)):
                results["errors"].append("Duplicate stage names detected")
                results["valid"] = False

            # Check for circular dependencies
            if self._has_circular_dependency():
                results["errors"].append("Circular dependency detected in pipeline")
                results["valid"] = False

            # Check all stages have models
            for stage in self._stages:
                if not stage.model:
                    results["errors"].append(f"Stage '{stage.name}' has no model")
                    results["valid"] = False

            # Check for unconnected required inputs
            connected_inputs = set()
            for conn in self._connections:
                connected_inputs.add((conn.target_stage, conn.target_input))

            for stage in self._stages:
                for inp in stage.inputs:
                    if (stage.name, inp) not in connected_inputs:
                        # This input needs to be provided at execution time
                        results["warnings"].append(
                            f"Input '{inp}' in stage '{stage.name}' must be "
                            "provided at execution time"
                        )

            if results["errors"]:
                raise PipelineValidationError(
                    f"Pipeline validation failed: {'; '.join(results['errors'])}"
                )

            return results

    def _has_circular_dependency(self) -> bool:
        """
        Check for circular dependencies in the pipeline.

        Returns:
            True if circular dependency exists, False otherwise.
        """
        # Build dependency graph
        graph: Dict[str, List[str]] = {s.name: [] for s in self._stages}

        for conn in self._connections:
            if conn.target_stage in graph:
                graph[conn.target_stage].append(conn.source_stage)

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(stage: str) -> bool:
            visited.add(stage)
            rec_stack.add(stage)

            for dep in graph.get(stage, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(stage)
            return False

        for stage_name in graph:
            if stage_name not in visited:
                if has_cycle(stage_name):
                    return True

        return False

    def get_execution_order(self) -> List[str]:
        """
        Get the topologically sorted execution order of stages.

        Returns:
            List of stage names in execution order.
        """
        with self._lock:
            # Build dependency graph
            in_degree: Dict[str, int] = {s.name: 0 for s in self._stages}
            graph: Dict[str, List[str]] = {s.name: [] for s in self._stages}

            for conn in self._connections:
                if conn.target_stage in in_degree:
                    in_degree[conn.target_stage] += 1
                if conn.source_stage in graph:
                    graph[conn.source_stage].append(conn.target_stage)

            # Kahn's algorithm for topological sort
            queue = [name for name, deg in in_degree.items() if deg == 0]
            result = []

            while queue:
                node = queue.pop(0)
                result.append(node)

                for neighbor in graph.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            return result

    def set_executor(self, executor: Callable) -> None:
        """
        Set a custom executor function for running stages.

        The executor should accept (stage, inputs) and return outputs dict.

        Args:
            executor: Callable that executes a stage.

        Raises:
            ValidationError: If executor is not callable.
        """
        if not callable(executor):
            raise ValidationError("executor must be callable")
        self._executor = executor

    def execute(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 300000
    ) -> PipelineResult:
        """
        Execute the pipeline with provided inputs.

        Args:
            inputs: Input data dictionary mapping input names to data.
            timeout_ms: Maximum execution time in milliseconds (must be > 0).

        Returns:
            PipelineResult with execution results.

        Raises:
            ValidationError: If timeout_ms is not a positive integer.
            PipelineExecutionError: If execution fails.
        """
        # Validate timeout_ms
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValidationError(f"timeout_ms must be a positive integer, got {timeout_ms}")

        start_time = time.time()
        inputs = inputs or {}

        with self._lock:
            # Validate before execution
            try:
                self.validate()
            except PipelineValidationError as e:
                return PipelineResult(
                    pipeline_name=self._name,
                    success=False,
                    metadata={"error": str(e)}
                )

            # Get execution order
            execution_order = self.get_execution_order()

            # Track data between stages
            stage_outputs: Dict[str, Dict[str, Any]] = {}
            stage_results: List[StageResult] = []
            all_success = True

            # Execute each stage
            for stage_name in execution_order:
                stage = self._stage_map[stage_name]

                if not stage.enabled:
                    continue

                # Gather inputs for this stage
                stage_inputs = {}

                # First, get inputs from pipeline inputs
                for inp in stage.inputs:
                    if inp in inputs:
                        stage_inputs[inp] = inputs[inp]

                # Then, get inputs from connected stages
                for conn in self._connections:
                    if conn.target_stage == stage_name:
                        if conn.source_stage in stage_outputs:
                            src_outputs = stage_outputs[conn.source_stage]
                            if conn.source_output in src_outputs:
                                stage_inputs[conn.target_input] = \
                                    src_outputs[conn.source_output]

                # Execute the stage
                stage_start = time.time()
                try:
                    if self._executor:
                        outputs = self._executor(stage, stage_inputs)
                    else:
                        # Default executor just passes through inputs
                        outputs = self._default_execute(stage, stage_inputs)

                    stage_outputs[stage_name] = outputs
                    stage_time_ms = (time.time() - stage_start) * 1000

                    stage_results.append(StageResult(
                        stage_name=stage_name,
                        success=True,
                        outputs=outputs,
                        execution_time_ms=stage_time_ms
                    ))

                except Exception as e:
                    stage_time_ms = (time.time() - stage_start) * 1000
                    all_success = False

                    stage_results.append(StageResult(
                        stage_name=stage_name,
                        success=False,
                        execution_time_ms=stage_time_ms,
                        error=str(e)
                    ))

                    # Stop execution on failure in sequential mode
                    if self._execution_strategy == "sequential":
                        break

            # Gather final outputs from the last stage(s)
            final_outputs = {}
            if execution_order and all_success:
                last_stage = execution_order[-1]
                if last_stage in stage_outputs:
                    final_outputs = stage_outputs[last_stage]

            total_time_ms = (time.time() - start_time) * 1000

            return PipelineResult(
                pipeline_name=self._name,
                success=all_success,
                stage_results=stage_results,
                final_outputs=final_outputs,
                total_time_ms=total_time_ms,
                metadata={
                    "execution_strategy": self._execution_strategy,
                    "stage_count": len(self._stages),
                    "executed_stages": len(stage_results),
                }
            )

    def _default_execute(
        self,
        stage: PipelineStage,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Default stage executor (passthrough for testing).

        Args:
            stage: Stage to execute.
            inputs: Input data for the stage.

        Returns:
            Output data dictionary.
        """
        # Default implementation creates placeholder outputs
        outputs = {}
        for output_name in stage.outputs:
            outputs[output_name] = {
                "stage": stage.name,
                "model": stage.model,
                "input_keys": list(inputs.keys()),
            }
        return outputs

    def clear(self) -> None:
        """Remove all stages and connections from the pipeline."""
        with self._lock:
            self._stages.clear()
            self._connections.clear()
            self._stage_map.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pipeline to dictionary for serialization.

        Returns:
            Dictionary representation of the pipeline.
        """
        with self._lock:
            return {
                "name": self._name,
                "description": self._description,
                "execution_strategy": self._execution_strategy,
                "created_at": self._created_at,
                "stages": [s.to_dict() for s in self._stages],
                "connections": [c.to_dict() for c in self._connections],
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """
        Create pipeline from dictionary.

        Args:
            data: Dictionary representation of a pipeline.

        Returns:
            Pipeline instance.
        """
        pipeline = cls(
            name=data.get("name"),
            description=data.get("description", ""),
            execution_strategy=data.get("execution_strategy", "sequential")
        )

        # Add stages
        for stage_data in data.get("stages", []):
            stage = PipelineStage.from_dict(stage_data)
            pipeline.add_stage(stage)

        # Add connections - use connect() for validation when possible,
        # fall back to direct append for backward compatibility with saved pipelines
        for conn_data in data.get("connections", []):
            conn = DataConnection.from_dict(conn_data)
            try:
                # Try to use connect() for validation
                pipeline.connect(
                    source_stage=conn.source_stage,
                    source_output=conn.source_output,
                    target_stage=conn.target_stage,
                    target_input=conn.target_input,
                    transform=conn.transform
                )
            except (StageNotFoundError, DataFlowError):
                # For backward compatibility, allow loading even if validation fails
                # The pipeline.validate() method will catch issues at execution time
                pipeline._connections.append(conn)

        return pipeline

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize pipeline to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Pipeline":
        """
        Create pipeline from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            Pipeline instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save pipeline to a JSON file.

        Args:
            path: Path to save the pipeline.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Pipeline":
        """
        Load pipeline from a JSON file.

        Args:
            path: Path to the pipeline file.

        Returns:
            Pipeline instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __len__(self) -> int:
        """Return number of stages."""
        return len(self._stages)

    def __contains__(self, stage_name: str) -> bool:
        """Check if stage exists."""
        return self.has_stage(stage_name)

    def __iter__(self):
        """Iterate over stages."""
        return iter(self._stages)

    def __repr__(self) -> str:
        return (
            f"Pipeline(name={self._name!r}, "
            f"stages={len(self._stages)}, "
            f"connections={len(self._connections)})"
        )

    # =========================================================================
    # Optimization Methods
    # =========================================================================

    def optimize(
        self,
        config: Optional[OptimizationConfig] = None
    ) -> OptimizationResult:
        """
        Apply optimizations to the pipeline.

        Applies various optimization techniques including quantization,
        operator fusion, and hardware-specific optimizations.

        Args:
            config: OptimizationConfig specifying which optimizations to apply.
                   If None, uses default configuration.

        Returns:
            OptimizationResult with details about applied optimizations.

        Example:
            config = OptimizationConfig(
                enable_quantization=True,
                quantization_bits=8,
                enable_fusion=True,
                use_fp16=True
            )
            result = pipeline.optimize(config)
        """
        config = config or OptimizationConfig()

        with self._lock:
            result = OptimizationResult(success=True)

            # Apply quantization if enabled
            if config.enable_quantization:
                quant_result = self._apply_quantization(config)
                result.quantized_stages.extend(quant_result["stages"])
                result.optimizations_applied.append(
                    f"quantization_{config.quantization_bits}bit"
                )
                result.memory_reduction *= quant_result.get("memory_factor", 1.0)

            # Apply operator fusion if enabled
            if config.enable_fusion:
                fusion_result = self._apply_fusion(config)
                result.fused_operations.extend(fusion_result["operations"])
                result.optimizations_applied.append("operator_fusion")
                result.estimated_speedup *= fusion_result.get("speedup_factor", 1.0)

            # Apply FP16 optimization if enabled
            if config.use_fp16:
                fp16_result = self._apply_fp16_optimization(config)
                result.optimizations_applied.append("fp16_acceleration")
                result.estimated_speedup *= fp16_result.get("speedup_factor", 1.0)

            # Apply SIMD group optimizations if enabled
            if config.use_simdgroup_operations:
                simd_result = self._apply_simdgroup_optimization(config)
                result.optimizations_applied.append("simdgroup_operations")
                result.estimated_speedup *= simd_result.get("speedup_factor", 1.0)

            # Store optimization config for later reference
            self._optimization_config = config
            self._optimization_result = result

            return result

    def _apply_quantization(
        self,
        config: OptimizationConfig
    ) -> Dict[str, Any]:
        """
        Apply quantization optimization to pipeline stages.

        Quantization reduces model precision from FP32 to INT8/INT16,
        reducing memory usage and potentially improving inference speed.

        Args:
            config: Optimization configuration.

        Returns:
            Dict with quantization results.
        """
        quantized_stages = []
        memory_factor = 1.0

        # Calculate memory reduction based on bit width
        if config.quantization_bits == 8:
            memory_factor = 0.25  # 32-bit to 8-bit = 4x reduction
        elif config.quantization_bits == 16:
            memory_factor = 0.5   # 32-bit to 16-bit = 2x reduction

        for stage in self._stages:
            if stage.enabled:
                # Mark stage as quantized in config
                stage.config["quantized"] = True
                stage.config["quantization_bits"] = config.quantization_bits
                stage.config["quantization_scheme"] = "symmetric"
                quantized_stages.append(stage.name)

        return {
            "stages": quantized_stages,
            "memory_factor": memory_factor,
            "bits": config.quantization_bits,
        }

    def _apply_fusion(
        self,
        config: OptimizationConfig
    ) -> Dict[str, Any]:
        """
        Apply operator fusion optimization.

        Fuses compatible consecutive operations to reduce memory
        bandwidth and kernel launch overhead.

        Args:
            config: Optimization configuration.

        Returns:
            Dict with fusion results.
        """
        fused_operations = []
        speedup_factor = 1.0

        # Analyze pipeline for fusable patterns
        execution_order = self.get_execution_order()

        for pattern in config.fusion_patterns:
            # Look for matching patterns in the pipeline
            matches = self._find_fusion_pattern(pattern, execution_order)
            if matches:
                fused_operations.extend(matches)
                # Estimate speedup from fusion (typically 10-20% per fusion)
                speedup_factor *= 1.15 ** len(matches)

        # Apply fusion markers to stages
        for i, stage_name in enumerate(execution_order[:-1]):
            next_stage = execution_order[i + 1]
            stage = self._stage_map[stage_name]
            next_stage_obj = self._stage_map[next_stage]

            # Check if these stages can be fused
            if self._can_fuse(stage, next_stage_obj):
                stage.config["fuse_with_next"] = next_stage
                fused_operations.append(f"{stage.model}+{next_stage_obj.model}")

        return {
            "operations": fused_operations,
            "speedup_factor": speedup_factor,
        }

    def _find_fusion_pattern(
        self,
        pattern: str,
        execution_order: List[str]
    ) -> List[str]:
        """
        Find instances of a fusion pattern in the pipeline.

        Args:
            pattern: Fusion pattern name (e.g., "conv_bn_relu").
            execution_order: List of stage names in execution order.

        Returns:
            List of matched fusion descriptions.
        """
        matches = []
        pattern_ops = pattern.split("_")

        # Sliding window search for pattern
        for i in range(len(execution_order) - len(pattern_ops) + 1):
            window = execution_order[i:i + len(pattern_ops)]
            window_models = [self._stage_map[s].model.lower() for s in window]

            # Check if window matches pattern
            if self._matches_pattern(window_models, pattern_ops):
                matches.append(f"{pattern}: {' -> '.join(window)}")

        return matches

    def _matches_pattern(
        self,
        models: List[str],
        pattern_ops: List[str]
    ) -> bool:
        """
        Check if a sequence of models matches a fusion pattern.

        Args:
            models: List of model names.
            pattern_ops: List of pattern operation types.

        Returns:
            True if models match the pattern.
        """
        if len(models) != len(pattern_ops):
            return False

        for model, op in zip(models, pattern_ops):
            if op not in model:
                return False

        return True

    def _can_fuse(
        self,
        stage1: PipelineStage,
        stage2: PipelineStage
    ) -> bool:
        """
        Check if two stages can be fused.

        Args:
            stage1: First pipeline stage.
            stage2: Second pipeline stage.

        Returns:
            True if stages can be fused.
        """
        # Check if there's a direct connection between stages
        for conn in self._connections:
            if (conn.source_stage == stage1.name and
                conn.target_stage == stage2.name):
                # Basic fusability check
                # In practice, would check operator types, shapes, etc.
                return True
        return False

    def _apply_fp16_optimization(
        self,
        config: OptimizationConfig
    ) -> Dict[str, Any]:
        """
        Apply FP16 (half-precision) optimization.

        Converts operations to use FP16 where supported for
        improved performance on modern GPUs.

        Args:
            config: Optimization configuration.

        Returns:
            Dict with FP16 optimization results.
        """
        fp16_stages = []

        for stage in self._stages:
            if stage.enabled:
                stage.config["precision"] = "fp16"
                stage.config["accumulator_type"] = "float16"
                fp16_stages.append(stage.name)

        return {
            "stages": fp16_stages,
            "speedup_factor": 1.5,  # Typical FP16 speedup on modern GPUs
        }

    def _apply_simdgroup_optimization(
        self,
        config: OptimizationConfig
    ) -> Dict[str, Any]:
        """
        Apply SIMD group optimization for Apple Silicon.

        Optimizes operations to use SIMD group (subgroup) operations
        for better performance on Apple Silicon GPUs.

        Args:
            config: Optimization configuration.

        Returns:
            Dict with SIMD group optimization results.
        """
        optimized_stages = []

        for stage in self._stages:
            if stage.enabled:
                stage.config["use_simdgroup"] = True
                stage.config["tile_size"] = config.tile_size
                stage.config["threadgroup_memory"] = config.threadgroup_memory
                optimized_stages.append(stage.name)

        return {
            "stages": optimized_stages,
            "speedup_factor": 1.2,  # Typical SIMD group speedup
        }

    def enable_quantization(
        self,
        bits: int = 8,
        scheme: str = "symmetric"
    ) -> "Pipeline":
        """
        Enable quantization for the pipeline.

        Convenience method to enable quantization optimization.

        Args:
            bits: Quantization bit width (8 or 16).
            scheme: Quantization scheme ("symmetric" or "asymmetric").

        Returns:
            Self for method chaining.
        """
        config = OptimizationConfig(
            enable_quantization=True,
            quantization_bits=bits
        )
        self.optimize(config)
        return self

    def enable_fusion(
        self,
        patterns: Optional[List[str]] = None
    ) -> "Pipeline":
        """
        Enable operator fusion for the pipeline.

        Convenience method to enable fusion optimization.

        Args:
            patterns: List of fusion patterns to apply.
                     If None, uses default patterns.

        Returns:
            Self for method chaining.
        """
        config = OptimizationConfig(enable_fusion=True)
        if patterns:
            config.fusion_patterns = patterns
        self.optimize(config)
        return self

    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get a report of applied optimizations.

        Returns:
            Dict with optimization details for each stage.
        """
        report = {
            "pipeline_name": self._name,
            "optimization_config": None,
            "optimization_result": None,
            "stage_details": [],
        }

        if self._optimization_config is not None:
            report["optimization_config"] = self._optimization_config.to_dict()

        if self._optimization_result is not None:
            report["optimization_result"] = self._optimization_result.to_dict()

        for stage in self._stages:
            stage_info = {
                "name": stage.name,
                "model": stage.model,
                "optimizations": {},
            }

            if stage.config.get("quantized"):
                stage_info["optimizations"]["quantization"] = {
                    "enabled": True,
                    "bits": stage.config.get("quantization_bits"),
                    "scheme": stage.config.get("quantization_scheme"),
                }

            if stage.config.get("precision") == "fp16":
                stage_info["optimizations"]["fp16"] = {
                    "enabled": True,
                    "accumulator": stage.config.get("accumulator_type"),
                }

            if stage.config.get("use_simdgroup"):
                stage_info["optimizations"]["simdgroup"] = {
                    "enabled": True,
                    "tile_size": stage.config.get("tile_size"),
                }

            if stage.config.get("fuse_with_next"):
                stage_info["optimizations"]["fusion"] = {
                    "fused_with": stage.config.get("fuse_with_next"),
                }

            report["stage_details"].append(stage_info)

        return report


def create_simple_pipeline(
    name: str,
    models: List[str],
    auto_connect: bool = True
) -> Pipeline:
    """
    Create a simple sequential pipeline from a list of models.

    Args:
        name: Name for the pipeline.
        models: List of model names in execution order.
        auto_connect: If True, automatically connect stages.

    Returns:
        Configured Pipeline instance.
    """
    pipeline = Pipeline(name=name)

    prev_stage_name = None
    prev_output = None

    for i, model in enumerate(models):
        stage_name = f"stage_{i}"
        input_name = f"input_{i}"
        output_name = f"output_{i}"

        stage = PipelineStage(
            name=stage_name,
            model=model,
            inputs=[input_name],
            outputs=[output_name]
        )
        pipeline.add_stage(stage)

        # Connect to previous stage
        if auto_connect and prev_stage_name and prev_output:
            pipeline.connect(
                source_stage=prev_stage_name,
                source_output=prev_output,
                target_stage=stage_name,
                target_input=input_name
            )

        prev_stage_name = stage_name
        prev_output = output_name

    return pipeline
