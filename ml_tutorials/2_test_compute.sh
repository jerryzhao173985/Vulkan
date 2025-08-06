#!/bin/bash
# Tutorial 2: Test Vulkan Compute Shaders

SDK="/Users/jerry/Vulkan/builds/ARM-ML-SDK-Complete"
export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK/lib

echo "=== Tutorial 2: Testing Compute Shaders ==="
echo ""

# Create a compute test
cat > /tmp/compute_test.json << 'EOF'
{
  "name": "Vector Addition Test",
  "description": "Add two vectors using Vulkan compute",
  "compute_operations": [
    {
      "shader": "add",
      "workgroup_size": [64, 1, 1],
      "dispatch": [1, 1, 1],
      "buffers": [
        {
          "name": "input_a",
          "size": 256,
          "data": "random"
        },
        {
          "name": "input_b", 
          "size": 256,
          "data": "random"
        },
        {
          "name": "output",
          "size": 256,
          "usage": "storage"
        }
      ]
    }
  ]
}
EOF

echo "Created compute test scenario"
echo ""
echo "What this does:"
echo "1. Creates 2 input buffers (256 floats each)"
echo "2. Loads 'add.spv' shader"
echo "3. Dispatches compute workgroup"
echo "4. Adds vectors element-wise"
echo "5. Stores result in output buffer"
echo ""

echo "Available compute shaders:"
ls $SDK/shaders/*.spv 2>/dev/null | head -5 | while read shader; do
    echo "  • $(basename $shader .spv)"
done
echo "  ... and 30 more"
echo ""

echo "To run: $SDK/bin/scenario-runner --scenario /tmp/compute_test.json"
echo ""
echo "Next: Run './ml_tutorials/3_benchmark.sh' to benchmark operations"