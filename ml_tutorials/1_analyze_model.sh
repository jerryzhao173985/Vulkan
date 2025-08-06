#!/bin/bash
# Tutorial 1: Analyze TensorFlow Lite Models

SDK="/Users/jerry/Vulkan/builds/ARM-ML-SDK-Complete"

echo "=== Tutorial 1: Analyzing ML Models ==="
echo ""

# Pick a model
MODEL="$SDK/models/mobilenet_v2_1.0_224_quantized_1_default_1.tflite"

echo "Model: MobileNet V2 (Quantized)"
echo "Size: $(du -h $MODEL | cut -f1)"
echo ""

# Use Python to analyze
python3 << EOF
import struct

def analyze_tflite(path):
    with open(path, 'rb') as f:
        # Read TFLite header
        data = f.read(8)
        if data[:4] == b'TFL3':
            print("✓ Valid TensorFlow Lite model")
            print("✓ Format version: TFL3")
        
        # Get file size
        f.seek(0, 2)
        size = f.tell()
        print(f"✓ Total size: {size:,} bytes")
        
        # Basic info
        print("\nModel Properties:")
        print("• Input: 224x224x3 image")
        print("• Output: 1001 classes")
        print("• Type: Quantized INT8")
        print("• Use: Image classification")

analyze_tflite("$MODEL")

print("\nHow to use this model:")
print("1. Load image (224x224 RGB)")
print("2. Normalize to [0,1]")
print("3. Quantize to INT8")
print("4. Run inference")
print("5. Get class predictions")
EOF

echo ""
echo "Next: Run './ml_tutorials/2_test_compute.sh' to test compute shaders"