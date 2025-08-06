#!/bin/bash
# Demo: Style Transfer with TensorFlow Lite Model

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SDK_ROOT="/Users/jerry/Vulkan"
SDK_BIN="$SDK_ROOT/builds/ARM-ML-SDK-Complete/bin"
MODELS="$SDK_ROOT/builds/ARM-ML-SDK-Complete/models"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        ARM ML SDK - Style Transfer Demo                   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Set up environment
export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK_ROOT/builds/ARM-ML-SDK-Complete/lib

echo -e "${CYAN}Available Style Models:${NC}"
echo "  1. La Muse"
echo "  2. Udnie"
echo "  3. Mirror"
echo "  4. Wave Crop"
echo "  5. Des Glaneuses"
echo ""

# Select model
read -p "Select style model (1-5): " choice

case $choice in
    1) MODEL="la_muse.tflite" ;;
    2) MODEL="udnie.tflite" ;;
    3) MODEL="mirror.tflite" ;;
    4) MODEL="wave_crop.tflite" ;;
    5) MODEL="des_glaneuses.tflite" ;;
    *) MODEL="la_muse.tflite" ;;
esac

echo ""
echo -e "${CYAN}Using model: $MODEL${NC}"
echo ""

# Check if scenario-runner exists
if [ ! -f "$SDK_BIN/scenario-runner" ]; then
    echo -e "${RED}Error: scenario-runner not found!${NC}"
    echo "Please build the SDK first: ./vulkan-ml-sdk-build build"
    exit 1
fi

# Create a sample scenario JSON
cat > /tmp/style_transfer_demo.json << EOF
{
  "name": "Style Transfer Demo",
  "description": "Apply artistic style transfer using $MODEL",
  "model": "$MODELS/$MODEL",
  "input": {
    "type": "image",
    "width": 256,
    "height": 256,
    "channels": 3
  },
  "output": {
    "type": "image",
    "width": 256,
    "height": 256,
    "channels": 3
  },
  "operations": [
    {
      "type": "style_transfer",
      "model": "$MODEL"
    }
  ]
}
EOF

echo -e "${CYAN}Running style transfer...${NC}"
echo ""

# Run the scenario
"$SDK_BIN/scenario-runner" \
    --scenario /tmp/style_transfer_demo.json \
    --output /tmp/style_output \
    --log-level info

echo ""
echo -e "${GREEN}✓ Style transfer complete!${NC}"
echo "Output saved to: /tmp/style_output"
echo ""

# Show performance metrics if available
if [ -f "/tmp/style_output/performance.json" ]; then
    echo -e "${CYAN}Performance Metrics:${NC}"
    cat /tmp/style_output/performance.json
fi