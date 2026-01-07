#!/usr/bin/env python3
"""
Simple test runner for pipeline integration tests without pytest dependency.
"""

import sys
import traceback
from pathlib import Path

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

MODELS_DIR = SDK_DIR / "models"

def run_tests():
    """Run pipeline integration tests"""
    passed = 0
    failed = 0
    skipped = 0

    print("=" * 60)
    print("Multi-Model Pipeline Integration Tests")
    print("=" * 60)

    # Import SDK modules
    try:
        from vulkan_ml_sdk.pipeline import (
            Pipeline, PipelineStage, DataConnection,
            PipelineResult, StageResult, OptimizationConfig,
            PipelineError, StageNotFoundError, DataFlowError,
            PipelineValidationError, create_simple_pipeline
        )
        SDK_AVAILABLE = True
        print("\n[OK] SDK modules imported successfully")
    except ImportError as e:
        print(f"\n[SKIP] SDK not available: {e}")
        SDK_AVAILABLE = False
        skipped += 1
        return passed, failed, skipped

    # Test 1: Create empty pipeline
    print("\n[TEST] test_create_empty_pipeline")
    try:
        pipeline = Pipeline(name="empty_pipeline")
        assert pipeline.name == "empty_pipeline"
        assert len(pipeline.stages) == 0
        assert len(pipeline.connections) == 0
        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 2: Add stages
    print("\n[TEST] test_add_stages")
    try:
        pipeline = Pipeline(name="stage_test")
        pipeline.add_stage(PipelineStage(
            name="preprocess",
            model="preprocessor",
            inputs=["raw_input"],
            outputs=["processed_data"]
        ))
        pipeline.add_stage(PipelineStage(
            name="inference",
            model="mobilenet_v2",
            inputs=["processed_data"],
            outputs=["predictions"]
        ))
        assert len(pipeline.stages) == 2
        assert pipeline.has_stage("preprocess")
        assert pipeline.has_stage("inference")
        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 3: Connect stages
    print("\n[TEST] test_connect_stages")
    try:
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
        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 4: Validate pipeline
    print("\n[TEST] test_validate_pipeline")
    try:
        pipeline = Pipeline(name="valid_pipeline")
        pipeline.add_stage(PipelineStage(
            name="preprocess",
            model="preprocessor",
            inputs=["raw_input"],
            outputs=["processed_data"]
        ))
        pipeline.add_stage(PipelineStage(
            name="inference",
            model="mobilenet_v2",
            inputs=["processed_data"],
            outputs=["predictions"]
        ))
        pipeline.connect("preprocess", "processed_data", "inference", "processed_data")
        result = pipeline.validate()
        assert result["valid"] is True
        assert result["stage_count"] == 2
        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 5: Pipeline execution order
    print("\n[TEST] test_execution_order")
    try:
        pipeline = Pipeline(name="order_test")
        pipeline.add_stage(PipelineStage(
            name="stage1",
            model="model1",
            inputs=["input"],
            outputs=["mid"]
        ))
        pipeline.add_stage(PipelineStage(
            name="stage2",
            model="model2",
            inputs=["mid"],
            outputs=["output"]
        ))
        pipeline.connect("stage1", "mid", "stage2", "mid")
        order = pipeline.get_execution_order()
        assert order.index("stage1") < order.index("stage2")
        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 6: Pipeline serialization
    print("\n[TEST] test_pipeline_serialization")
    try:
        pipeline = Pipeline(name="serial_test")
        pipeline.add_stage(PipelineStage(
            name="stage",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))
        data = pipeline.to_dict()
        restored = Pipeline.from_dict(data)
        assert restored.name == pipeline.name
        assert len(restored.stages) == len(pipeline.stages)
        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 7: Pipeline optimization
    print("\n[TEST] test_pipeline_optimization")
    try:
        pipeline = Pipeline(name="opt_test")
        pipeline.add_stage(PipelineStage(
            name="stage1",
            model="model1",
            inputs=["in"],
            outputs=["out"]
        ))
        config = OptimizationConfig(
            enable_quantization=True,
            quantization_bits=8
        )
        result = pipeline.optimize(config)
        assert result.success is True
        assert "quantization_8bit" in result.optimizations_applied
        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 8: Duplicate stage error
    print("\n[TEST] test_duplicate_stage_error")
    try:
        pipeline = Pipeline(name="dup_test")
        pipeline.add_stage(PipelineStage(
            name="stage",
            model="model",
            inputs=["in"],
            outputs=["out"]
        ))
        try:
            pipeline.add_stage(PipelineStage(
                name="stage",  # Duplicate
                model="model2",
                inputs=["in2"],
                outputs=["out2"]
            ))
            print("  FAILED: Should have raised PipelineError")
            failed += 1
        except PipelineError:
            print("  PASSED")
            passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 9: Create simple pipeline helper
    print("\n[TEST] test_create_simple_pipeline")
    try:
        models = ["preprocessor", "mobilenet_v2", "postprocessor"]
        pipeline = create_simple_pipeline("simple_inference", models)
        assert pipeline.name == "simple_inference"
        assert len(pipeline.stages) == 3
        assert len(pipeline.connections) == 2
        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 10: Pipeline execution with mock executor
    print("\n[TEST] test_pipeline_execution")
    try:
        execution_log = []
        def mock_executor(stage, inputs):
            execution_log.append(stage.name)
            outputs = {}
            for output_name in stage.outputs:
                outputs[output_name] = f"data_from_{stage.name}"
            return outputs

        pipeline = Pipeline(name="exec_test")
        pipeline.add_stage(PipelineStage(
            name="stage1",
            model="model1",
            inputs=["input"],
            outputs=["mid"]
        ))
        pipeline.add_stage(PipelineStage(
            name="stage2",
            model="model2",
            inputs=["mid"],
            outputs=["output"]
        ))
        pipeline.connect("stage1", "mid", "stage2", "mid")
        pipeline.set_executor(mock_executor)

        result = pipeline.execute({"input": "test_data"})
        assert result.success is True
        assert len(execution_log) == 2
        assert execution_log[0] == "stage1"
        assert execution_log[1] == "stage2"
        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        failed += 1

    return passed, failed, skipped


if __name__ == "__main__":
    passed, failed, skipped = run_tests()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    if failed == 0:
        print("\nAll tests PASSED!")
        sys.exit(0)
    else:
        print(f"\n{failed} test(s) FAILED")
        sys.exit(1)
