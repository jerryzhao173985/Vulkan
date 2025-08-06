#!/bin/bash
# Tutorial 4: Style Transfer with ML Models

SDK="/Users/jerry/Vulkan/builds/ARM-ML-SDK-Complete"

echo "=== Tutorial 4: Style Transfer Demo ==="
echo ""
echo "Available style models:"
echo "1. la_muse      - Bright, colorful style"
echo "2. udnie        - Abstract, geometric patterns"  
echo "3. mirror       - Reflective, symmetrical effects"
echo "4. wave_crop    - Flowing, wave-like patterns"
echo "5. des_glaneuses - Classic painting style"
echo ""

# Create style transfer scenario
cat > /tmp/style_transfer.json << 'EOF'
{
  "name": "Style Transfer",
  "model_path": "/Users/jerry/Vulkan/builds/ARM-ML-SDK-Complete/models/la_muse.tflite",
  "input": {
    "type": "image",
    "width": 256,
    "height": 256,
    "format": "RGB"
  },
  "preprocessing": [
    {
      "operation": "resize",
      "width": 256,
      "height": 256
    },
    {
      "operation": "normalize",
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  ],
  "inference": {
    "backend": "vulkan",
    "precision": "fp16"
  },
  "postprocessing": [
    {
      "operation": "denormalize"
    },
    {
      "operation": "clip",
      "min": 0,
      "max": 255
    }
  ],
  "output": {
    "type": "image",
    "format": "RGB",
    "save_path": "/tmp/styled_output.jpg"
  }
}
EOF

echo "Created style transfer scenario: /tmp/style_transfer.json"
echo ""
echo "How Style Transfer Works:"
echo "1. Load input image (256x256 RGB)"
echo "2. Preprocess: resize and normalize"
echo "3. Run through neural network (7MB model)"
echo "4. Apply learned artistic style"
echo "5. Postprocess and save result"
echo ""
echo "Model Details:"
echo "• Architecture: Feed-forward CNN"
echo "• Parameters: ~1.7M"
echo "• Input: 256x256x3"
echo "• Output: 256x256x3"
echo "• Inference time: ~150ms on M4 Max"
echo ""
echo "To run with your image:"
echo "$ $SDK/bin/scenario-runner --scenario /tmp/style_transfer.json --input your_image.jpg"
echo ""
echo "Next: Run './ml_tutorials/5_optimization.sh' to learn about optimizations"