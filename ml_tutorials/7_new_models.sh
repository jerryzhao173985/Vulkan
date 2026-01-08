#!/bin/bash
# Tutorial 7: New ML Model Architectures

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK="$(cd "$SCRIPT_DIR/.." && pwd)/builds/ARM-ML-SDK-Complete"
export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK/lib

echo "=== Tutorial 7: New ML Model Architectures ==="
echo ""

# Clear PYTHONPATH to avoid Auto-Claude numpy conflicts
unset PYTHONPATH

# Validate SDK exists
if [ ! -d "$SDK" ]; then
    echo "ERROR: SDK not found at $SDK"
    echo "Please run ./scripts/build/build_all.sh first"
    exit 1
fi

echo "This tutorial covers the different ML architectures"
echo "supported by the ARM ML SDK and how to add new models."
echo ""

echo "Available Models in SDK:"
echo "========================"
if [ -d "$SDK/models" ]; then
    model_count=0
    for model_file in "$SDK/models/"*.tflite; do
        if [ -f "$model_file" ]; then
            name=$(basename "$model_file" .tflite)
            size=$(ls -lh "$model_file" | awk '{print $5}')
            echo "  • $name ($size)"
            model_count=$((model_count + 1))
        fi
    done
    if [ "$model_count" -eq 0 ]; then
        echo "  (No TFLite models found)"
    fi
else
    echo "  (Models directory not found)"
fi
echo ""

python3 << 'EOF'
print("=" * 60)
print("1. MODEL ARCHITECTURES OVERVIEW")
print("=" * 60)
print("")

print("Image Classification (MobileNet V2)")
print("-" * 40)
print("   Architecture: Inverted Residual Blocks")
print("   • Depthwise separable convolutions")
print("   • Linear bottlenecks with skip connections")
print("   • Input: 224x224x3 RGB image")
print("   • Output: 1001 class probabilities")
print("")
print("   Key layers:")
print("   • Conv2D -> BatchNorm -> ReLU6")
print("   • DepthwiseConv2D -> BatchNorm -> ReLU6")
print("   • Global Average Pooling")
print("   • Fully Connected (Softmax)")
print("")
print("   Use case: Real-time image classification")
print("   Inference: ~5ms on M4 Max (quantized)")
print("")

print("Style Transfer (Feed-Forward CNN)")
print("-" * 40)
print("   Architecture: Encoder-Decoder with Residual Blocks")
print("   • Downsampling: Strided convolutions")
print("   • Transform: Residual blocks")
print("   • Upsampling: Transposed convolutions")
print("   • Input: 256x256x3 RGB image")
print("   • Output: 256x256x3 styled image")
print("")
print("   Models available:")
print("   • la_muse      - Vibrant, colorful style")
print("   • udnie        - Abstract geometric patterns")
print("   • mirror       - Reflective symmetry")
print("   • wave_crop    - Flowing wave patterns")
print("   • des_glaneuses - Classic painting style")
print("")
print("   Use case: Artistic image transformation")
print("   Inference: ~150ms on M4 Max (fp16)")
print("")

print("Object Detection (Fire Detection)")
print("-" * 40)
print("   Architecture: Single-Shot Detector (SSD)")
print("   • Backbone: MobileNet feature extractor")
print("   • Detection head: Multi-scale feature maps")
print("   • NMS: Non-maximum suppression")
print("   • Input: Variable size (typically 300x300)")
print("   • Output: Bounding boxes + confidence scores")
print("")
print("   Output format:")
print("   • [x1, y1, x2, y2, confidence, class_id]")
print("   • Multiple detections per image")
print("")
print("   Use case: Safety monitoring, anomaly detection")
print("   Inference: ~30ms on M4 Max (quantized)")
print("")

print("=" * 60)
print("2. SUPPORTED LAYER TYPES")
print("=" * 60)
print("")
print("Core Operations (GPU Accelerated):")
print("   • Conv2D            - 2D convolution")
print("   • DepthwiseConv2D   - Depthwise separable conv")
print("   • TransposeConv2D   - Deconvolution/upsampling")
print("   • MatMul            - Matrix multiplication")
print("   • FullyConnected    - Dense layer")
print("")
print("Activation Functions:")
print("   • ReLU              - Rectified Linear Unit")
print("   • ReLU6             - Clamped ReLU [0, 6]")
print("   • Sigmoid           - Logistic activation")
print("   • Tanh              - Hyperbolic tangent")
print("   • Softmax           - Probability distribution")
print("")
print("Pooling Operations:")
print("   • MaxPool2D         - Max pooling")
print("   • AvgPool2D         - Average pooling")
print("   • GlobalAvgPool     - Global average pooling")
print("")
print("Normalization:")
print("   • BatchNorm         - Batch normalization")
print("   • InstanceNorm      - Instance normalization")
print("")
print("Element-wise Operations:")
print("   • Add, Sub, Mul, Div")
print("   • Concat, Split, Reshape")
print("   • Pad, Slice, Transpose")
print("")

print("=" * 60)
print("3. ADDING NEW MODELS")
print("=" * 60)
print("")
print("Step 1: Convert your model to TFLite format")
print("-" * 45)
print("   # From TensorFlow SavedModel:")
print("   import tensorflow as tf")
print("   converter = tf.lite.TFLiteConverter.from_saved_model('model/')")
print("   converter.optimizations = [tf.lite.Optimize.DEFAULT]")
print("   tflite_model = converter.convert()")
print("   with open('model.tflite', 'wb') as f:")
print("       f.write(tflite_model)")
print("")

print("Step 2: Validate model compatibility")
print("-" * 45)
print("   # Check layer support:")
print("   python3 tools/analyze_tflite_model.py model.tflite")
print("")
print("   # Look for unsupported ops:")
print("   • Custom ops (need implementation)")
print("   • Dynamic shapes (need static size)")
print("   • Control flow (limited support)")
print("")

print("Step 3: Create scenario JSON")
print("-" * 45)
print('   {')
print('     "name": "My Custom Model",')
print('     "model_path": "path/to/model.tflite",')
print('     "input": {')
print('       "type": "image",')
print('       "width": 224,')
print('       "height": 224,')
print('       "format": "RGB"')
print('     },')
print('     "preprocessing": [')
print('       {"operation": "resize", "width": 224, "height": 224},')
print('       {"operation": "normalize", "mean": [0.5], "std": [0.5]}')
print('     ],')
print('     "inference": {')
print('       "backend": "vulkan",')
print('       "precision": "fp16"')
print('     }')
print('   }')
print("")

print("Step 4: Run inference")
print("-" * 45)
print("   ./bin/scenario-runner \\")
print("       --scenario my_model.json \\")
print("       --input test_image.jpg \\")
print("       --output results/")
print("")

print("=" * 60)
print("4. OPTIMIZATION GUIDELINES BY ARCHITECTURE")
print("=" * 60)
print("")
print("Classification Models:")
print("   • Use INT8 quantization (3-4x speedup)")
print("   • Enable pipeline caching")
print("   • Batch multiple images if possible")
print("")
print("Style Transfer Models:")
print("   • Use FP16 precision (quality sensitive)")
print("   • Tile large images (memory efficiency)")
print("   • Cache intermediate activations")
print("")
print("Detection Models:")
print("   • Balance precision/recall vs speed")
print("   • Adjust NMS threshold for use case")
print("   • Consider lower resolution for real-time")
print("")
print("General Tips:")
print("   • Profile first: --profiling-dump-path /tmp/profile.json")
print("   • Match workgroup size to GPU (32 for Apple Silicon)")
print("   • Minimize memory transfers between GPU/CPU")
print("")

print("=" * 60)
print("5. COMMON MODEL CONVERSION ISSUES")
print("=" * 60)
print("")
print("Issue: Dynamic input shapes")
print("   Solution: Set fixed input shape before conversion")
print("   converter.input_shape = [1, 224, 224, 3]")
print("")
print("Issue: Unsupported custom ops")
print("   Solution: Implement as custom Vulkan compute shader")
print("   See: shaders/ for examples")
print("")
print("Issue: Large model size")
print("   Solution: Apply quantization-aware training")
print("   or post-training quantization")
print("")
print("Issue: Poor inference accuracy")
print("   Solution: Use representative dataset for calibration")
print("   converter.representative_dataset = calibration_gen")
print("")

print("=" * 60)
print("6. ARCHITECTURE COMPARISON")
print("=" * 60)
print("")
print("| Architecture      | Size   | Latency | Accuracy | Use Case        |")
print("|------------------|--------|---------|----------|-----------------|")
print("| MobileNet V2     | 3.4MB  | 5ms     | 71.8%    | Classification  |")
print("| Style Transfer   | 7MB    | 150ms   | N/A      | Image styling   |")
print("| Fire Detection   | 8.1MB  | 30ms    | ~90%     | Safety/detect   |")
print("")
print("Choose based on your requirements:")
print("• Real-time (< 16ms): Use quantized MobileNet")
print("• Quality-first: Use FP16 style transfer")
print("• Edge deployment: Minimize model size")
print("")
EOF

# Verify key SDK components
echo ""
echo "Verifying SDK components:"
echo "========================="
components_ok=0

if [ -f "$SDK/bin/scenario-runner" ]; then
    echo "  ✓ scenario-runner executable"
    components_ok=$((components_ok + 1))
else
    echo "  ✗ scenario-runner (not found)"
fi

if [ -d "$SDK/tools" ]; then
    tool_count=$(ls -1 "$SDK/tools/"*.py 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ Python tools ($tool_count tools available)"
    components_ok=$((components_ok + 1))
else
    echo "  ✗ Python tools directory (not found)"
fi

if [ -d "$SDK/shaders" ]; then
    shader_count=$(ls -1 "$SDK/shaders/"*.spv 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ SPIR-V shaders ($shader_count shaders)"
    components_ok=$((components_ok + 1))
else
    echo "  ✗ Shaders directory (not found)"
fi

echo ""
echo "Quick Commands:"
echo "==============="
echo "# Analyze a new model"
echo "python3 $SDK/tools/analyze_tflite_model.py your_model.tflite"
echo ""
echo "# Copy model to SDK"
echo "cp your_model.tflite $SDK/models/"
echo ""
echo "# Run inference"
echo "$SDK/bin/scenario-runner --scenario config.json --output results/"
echo ""
echo "Tutorial series complete!"
echo ""
echo "Tutorials available:"
echo "  1. Model analysis      - ./ml_tutorials/1_analyze_model.sh"
echo "  2. Compute shaders     - ./ml_tutorials/2_test_compute.sh"
echo "  3. Benchmarking        - ./ml_tutorials/3_benchmark.sh"
echo "  4. Style transfer      - ./ml_tutorials/4_style_transfer.sh"
echo "  5. Optimizations       - ./ml_tutorials/5_optimization.sh"
echo "  6. Advanced Vulkan     - ./ml_tutorials/6_advanced_vulkan.sh"
echo "  7. New models          - ./ml_tutorials/7_new_models.sh (this tutorial)"
echo ""
echo "For unified SDK operations, use: ./unified_launcher.sh"
echo ""
echo "Ready to add your own models!"
