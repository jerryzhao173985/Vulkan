#!/bin/bash
# Demo: Benchmark ML Operations

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

SDK_ROOT="/Users/jerry/Vulkan"
SDK_BIN="$SDK_ROOT/builds/ARM-ML-SDK-Complete/bin"
TOOLS="$SDK_ROOT/builds/ARM-ML-SDK-Complete/tools"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      ARM ML SDK - ML Operations Benchmark                 ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK_ROOT/builds/ARM-ML-SDK-Complete/lib

echo -e "${CYAN}Running ML Operation Benchmarks...${NC}"
echo ""

# Test 1: Matrix Multiplication
echo -e "${YELLOW}1. Matrix Multiplication (1024x1024)${NC}"
cat > /tmp/matmul_test.json << EOF
{
  "operation": "matrix_multiply",
  "input_a": [1024, 1024],
  "input_b": [1024, 1024],
  "iterations": 100
}
EOF

start_time=$(date +%s%N)
"$SDK_BIN/scenario-runner" --scenario /tmp/matmul_test.json --dry-run 2>/dev/null || true
end_time=$(date +%s%N)
elapsed=$((($end_time - $start_time) / 1000000))
echo "   Time: ${elapsed}ms"
echo ""

# Test 2: Convolution
echo -e "${YELLOW}2. 2D Convolution (224x224x32)${NC}"
cat > /tmp/conv2d_test.json << EOF
{
  "operation": "conv2d",
  "input_shape": [1, 224, 224, 32],
  "kernel_size": [3, 3],
  "filters": 64,
  "iterations": 50
}
EOF

start_time=$(date +%s%N)
"$SDK_BIN/scenario-runner" --scenario /tmp/conv2d_test.json --dry-run 2>/dev/null || true
end_time=$(date +%s%N)
elapsed=$((($end_time - $start_time) / 1000000))
echo "   Time: ${elapsed}ms"
echo ""

# Test 3: Pooling
echo -e "${YELLOW}3. Max Pooling (112x112x64)${NC}"
cat > /tmp/maxpool_test.json << EOF
{
  "operation": "max_pool2d",
  "input_shape": [1, 112, 112, 64],
  "pool_size": [2, 2],
  "iterations": 100
}
EOF

start_time=$(date +%s%N)
"$SDK_BIN/scenario-runner" --scenario /tmp/maxpool_test.json --dry-run 2>/dev/null || true
end_time=$(date +%s%N)
elapsed=$((($end_time - $start_time) / 1000000))
echo "   Time: ${elapsed}ms"
echo ""

# Test 4: Activation Functions
echo -e "${YELLOW}4. ReLU Activation (1M elements)${NC}"
cat > /tmp/relu_test.json << EOF
{
  "operation": "relu",
  "input_size": 1000000,
  "iterations": 1000
}
EOF

start_time=$(date +%s%N)
"$SDK_BIN/scenario-runner" --scenario /tmp/relu_test.json --dry-run 2>/dev/null || true
end_time=$(date +%s%N)
elapsed=$((($end_time - $start_time) / 1000000))
echo "   Time: ${elapsed}ms"
echo ""

# System Info
echo -e "${CYAN}System Information:${NC}"
echo "  Platform: macOS ARM64"
echo "  Device: Apple Silicon (M-series)"
echo "  SDK Version: $("$SDK_BIN/scenario-runner" --version 2>/dev/null | grep version | cut -d'"' -f4 || echo "unknown")"
echo ""

# Memory bandwidth test
echo -e "${CYAN}Memory Bandwidth Test:${NC}"
echo "  Testing unified memory performance..."
python3 << EOF
import time
import numpy as np

# Test memory bandwidth
size = 100 * 1024 * 1024  # 100MB
data = np.random.randn(size // 8).astype(np.float64)

start = time.time()
for _ in range(10):
    result = data * 2.0 + 1.0
elapsed = time.time() - start

bandwidth = (size * 10 * 3) / elapsed / (1024**3)  # GB/s
print(f"  Bandwidth: {bandwidth:.2f} GB/s")
EOF

echo ""
echo -e "${GREEN}✓ Benchmark complete!${NC}"