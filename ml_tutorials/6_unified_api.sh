#!/bin/bash
# Tutorial 6: Unified Python API - 3-Line Inference

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK="$(cd "$SCRIPT_DIR/.." && pwd)/builds/ARM-ML-SDK-Complete"
export PYTHONPATH="$SDK/lib/python:$PYTHONPATH"
export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK/lib

echo "=== Tutorial 6: Unified Python API ==="
echo ""

echo "The vulkan_ml_sdk package provides a simple 3-line inference API:"
echo ""
echo "  import vulkan_ml_sdk as vms"
echo "  sdk = vms.SDK()"
echo "  result = sdk.classify(image)"
echo ""

python3 << 'EOF'
import sys
sys.path.insert(0, sys.argv[0] if len(sys.argv) > 0 else ".")

print("=" * 50)
print("1. SDK INITIALIZATION")
print("=" * 50)
print("")
print("The SDK auto-detects your environment:")
print("")
print("  import vulkan_ml_sdk as vms")
print("  sdk = vms.SDK()")
print("")
print("What happens:")
print("• Auto-discovers SDK installation path")
print("• Sets up DYLD_LIBRARY_PATH for Vulkan/MoltenVK")
print("• Validates scenario-runner is available")
print("• Initializes model registry and cache")
print("")

print("=" * 50)
print("2. THREE-LINE INFERENCE")
print("=" * 50)
print("")
print("Classification (3 lines):")
print("")
print("  import vulkan_ml_sdk as vms")
print("  sdk = vms.SDK()")
print("  result = sdk.classify('/path/to/image.jpg')")
print("")
print("Style Transfer (3 lines):")
print("")
print("  import vulkan_ml_sdk as vms")
print("  sdk = vms.SDK()")
print("  result = sdk.style_transfer(image, style='la_muse')")
print("")

print("=" * 50)
print("3. MODEL LOADING & REUSE")
print("=" * 50)
print("")
print("For repeated inference, load the model once:")
print("")
print("  sdk = vms.SDK()")
print("  model = sdk.load_model('mobilenet_v2')")
print("")
print("  # Run inference multiple times")
print("  for image in images:")
print("      result = model.infer({'image': image})")
print("      print(result.outputs)")
print("")
print("Benefits:")
print("• Model loaded once, cached in memory")
print("• ~10x faster for batch processing")
print("• Automatic warm-start on subsequent runs")
print("")

print("=" * 50)
print("4. MULTI-MODEL PIPELINES")
print("=" * 50)
print("")
print("Chain multiple models together:")
print("")
print("  sdk = vms.SDK()")
print("  pipeline = sdk.create_simple_pipeline(")
print("      name='style_and_classify',")
print("      models=['la_muse', 'mobilenet_v2']")
print("  )")
print("  result = sdk.execute_pipeline(pipeline, inputs)")
print("")
print("Or build custom pipelines:")
print("")
print("  from vulkan_ml_sdk import Pipeline, PipelineStage")
print("")
print("  pipeline = Pipeline(name='custom')")
print("  pipeline.add_stage(PipelineStage(")
print("      name='preprocess',")
print("      model='preprocessor',")
print("      inputs=['raw_image'],")
print("      outputs=['normalized']")
print("  ))")
print("")

print("=" * 50)
print("5. CONVENIENCE FUNCTIONS")
print("=" * 50)
print("")
print("Module-level functions for even simpler usage:")
print("")
print("  import vulkan_ml_sdk as vms")
print("")
print("  # One-liner classification")
print("  result = vms.classify('photo.jpg')")
print("  print(f'Class: {result[\"top_class\"]}')")
print("  print(f'Confidence: {result[\"confidence\"]:.2%}')")
print("")
print("  # One-liner style transfer")
print("  result = vms.style_transfer('photo.jpg', style='udnie')")
print("  styled = result['stylized_image']")
print("")

print("=" * 50)
print("6. AVAILABLE MODELS")
print("=" * 50)
print("")
print("Classification:")
print("• mobilenet_v2 - General image classification (1001 classes)")
print("• fire_detection - Fire/smoke detection")
print("")
print("Style Transfer:")
print("• la_muse - Pablo Picasso inspired")
print("• udnie - Francis Picabia inspired")
print("• mirror - Abstract mirror style")
print("• wave_crop - The Great Wave inspired")
print("• des_glaneuses - Jean-François Millet inspired")
print("")

print("=" * 50)
print("7. WORKING WITH RESULTS")
print("=" * 50)
print("")
print("Classification results:")
print("")
print("  result = sdk.classify(image)")
print("  print(result['top_class'])      # Predicted class ID")
print("  print(result['confidence'])     # Confidence score (0-1)")
print("  print(result['predictions'])    # Top-k predictions list")
print("  print(result['execution_time_ms'])  # Inference time")
print("")
print("Style transfer results:")
print("")
print("  result = sdk.style_transfer(image, style='la_muse')")
print("  styled = result['stylized_image']  # numpy array")
print("  print(result['style'])             # Style used")
print("  print(result['output_size'])       # Output dimensions")
print("  print(result['execution_time_ms']) # Inference time")
print("")

print("=" * 50)
print("8. ERROR HANDLING")
print("=" * 50)
print("")
print("Robust error handling built-in:")
print("")
print("  from vulkan_ml_sdk import SDKError, InferenceError")
print("")
print("  try:")
print("      result = sdk.classify('image.jpg')")
print("  except SDKError as e:")
print("      print(f'SDK error: {e}')")
print("  except InferenceError as e:")
print("      print(f'Inference failed: {e}')")
print("")

print("=" * 50)
print("QUICK REFERENCE")
print("=" * 50)
print("")
print("# Import")
print("import vulkan_ml_sdk as vms")
print("")
print("# Initialize")
print("sdk = vms.SDK()")
print("")
print("# Classify image")
print("result = sdk.classify(image)")
print("")
print("# Style transfer")
print("result = sdk.style_transfer(image, style='la_muse')")
print("")
print("# Load model for repeated use")
print("model = sdk.load_model('mobilenet_v2')")
print("result = model.infer(inputs)")
print("")
print("# Create pipeline")
print("pipeline = sdk.create_simple_pipeline('name', ['model1', 'model2'])")
print("result = sdk.execute_pipeline(pipeline, inputs)")
print("")
EOF

echo ""
echo "API Package Location: $SDK/lib/python/vulkan_ml_sdk/"
echo ""
echo "To use in your Python scripts:"
echo "  export PYTHONPATH=$SDK/lib/python:\$PYTHONPATH"
echo "  python3 your_script.py"
echo ""
echo "All tutorials complete! Full API summary:"
echo "✓ Tutorial 1: Model analysis"
echo "✓ Tutorial 2: Compute shaders"
echo "✓ Tutorial 3: Benchmarking"
echo "✓ Tutorial 4: Style transfer"
echo "✓ Tutorial 5: Apple Silicon optimizations"
echo "✓ Tutorial 6: Unified Python API (3-line inference)"
echo ""
echo "Start building with: import vulkan_ml_sdk as vms"
