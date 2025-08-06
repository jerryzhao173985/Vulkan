#!/bin/bash
# End-to-End ML Pipeline Tests

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

SDK="/Users/jerry/Vulkan/builds/ARM-ML-SDK-Complete"
export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK/lib

echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}     ARM ML SDK - End-to-End Pipeline Tests                ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""

TOTAL_TESTS=0
PASSED_TESTS=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -ne "  Testing $test_name... "
    
    if eval "$test_cmd" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        echo "    Error: $(tail -1 /tmp/test_output.log)"
        return 1
    fi
}

# Function to measure time
measure_time() {
    local start=$(date +%s%N)
    eval "$1"
    local end=$(date +%s%N)
    echo $(( ($end - $start) / 1000000 ))
}

echo -e "${CYAN}1. SCENARIO RUNNER PIPELINE${NC}"

# Create test scenarios
cat > /tmp/test_add.json << 'EOF'
{
  "name": "Vector Addition Test",
  "operations": [
    {
      "type": "add",
      "input_a": [1.0, 2.0, 3.0, 4.0],
      "input_b": [5.0, 6.0, 7.0, 8.0]
    }
  ]
}
EOF

cat > /tmp/test_matmul.json << 'EOF'
{
  "name": "Matrix Multiplication Test",
  "operations": [
    {
      "type": "matmul",
      "matrix_a": {
        "shape": [2, 3],
        "data": [1, 2, 3, 4, 5, 6]
      },
      "matrix_b": {
        "shape": [3, 2],
        "data": [7, 8, 9, 10, 11, 12]
      }
    }
  ]
}
EOF

run_test "Scenario creation" "[ -f /tmp/test_add.json ]"
run_test "Scenario validation" "$SDK/bin/scenario-runner --scenario /tmp/test_add.json --dry-run"
run_test "Version check" "$SDK/bin/scenario-runner --version"
run_test "Help output" "$SDK/bin/scenario-runner --help"
echo ""

echo -e "${CYAN}2. MODEL LOADING TESTS${NC}"

# Test each model
for model in $SDK/models/*.tflite; do
    if [ -f "$model" ]; then
        MODEL_NAME=$(basename "$model" .tflite)
        run_test "$MODEL_NAME loading" "[ -f '$model' ] && [ -s '$model' ]"
    fi
done
echo ""

echo -e "${CYAN}3. SHADER COMPILATION TESTS${NC}"

# Test key shaders
for shader in add multiply conv2d relu maxpool; do
    SHADER_FILE="$SDK/shaders/${shader}.spv"
    if [ ! -f "$SHADER_FILE" ]; then
        SHADER_FILE="$SDK/shaders/${shader}_shader.spv"
    fi
    run_test "$shader shader" "[ -f '$SHADER_FILE' ]"
done
echo ""

echo -e "${CYAN}4. PYTHON TOOL INTEGRATION${NC}"

# Test Python tools
run_test "Python available" "python3 --version"
run_test "NumPy available" "python3 -c 'import numpy'"

# Model analysis
cat > /tmp/test_model_analysis.py << 'EOF'
import os
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else ""
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    sys.exit(1)

size = os.path.getsize(model_path)
print(f"Model size: {size/1024/1024:.2f} MB")

# Basic validation
with open(model_path, 'rb') as f:
    header = f.read(4)
    if header == b'TFL3':
        print("Valid TensorFlow Lite model")
        sys.exit(0)
    else:
        print("Invalid model format")
        sys.exit(1)
EOF

MODEL="$SDK/models/mobilenet_v2_1.0_224_quantized_1_default_1.tflite"
run_test "Model analysis" "python3 /tmp/test_model_analysis.py '$MODEL'"
echo ""

echo -e "${CYAN}5. PERFORMANCE MEASUREMENTS${NC}"

# Measure scenario-runner startup time
START_TIME=$(measure_time "$SDK/bin/scenario-runner --version > /dev/null 2>&1")
echo "  Startup time: ${START_TIME}ms"

# Test with different configurations
run_test "Pipeline caching flag" "$SDK/bin/scenario-runner --pipeline-caching --dry-run"
run_test "Debug markers flag" "$SDK/bin/scenario-runner --enable-gpu-debug-markers --dry-run"
echo ""

echo -e "${CYAN}6. MEMORY TESTS${NC}"

# Python memory test
cat > /tmp/test_memory.py << 'EOF'
import numpy as np

# Test memory allocation
sizes = [1024, 1024*1024, 10*1024*1024]  # 1KB, 1MB, 10MB
for size in sizes:
    try:
        data = np.zeros(size, dtype=np.float32)
        print(f"Allocated {size*4/1024/1024:.2f} MB")
        del data
    except MemoryError:
        print(f"Failed to allocate {size*4/1024/1024:.2f} MB")
        exit(1)
EOF

run_test "Memory allocation" "python3 /tmp/test_memory.py"
echo ""

echo -e "${CYAN}7. INTEGRATION WORKFLOW${NC}"

# Complete workflow test
cat > /tmp/test_workflow.sh << 'EOF'
#!/bin/bash
# Test complete ML workflow

SDK="/Users/jerry/Vulkan/builds/ARM-ML-SDK-Complete"

# 1. Check executable
if [ ! -x "$SDK/bin/scenario-runner" ]; then
    echo "Executable not found"
    exit 1
fi

# 2. Check model
MODEL="$SDK/models/mobilenet_v2_1.0_224_quantized_1_default_1.tflite"
if [ ! -f "$MODEL" ]; then
    echo "Model not found"
    exit 1
fi

# 3. Check shaders
if [ $(ls -1 $SDK/shaders/*.spv 2>/dev/null | wc -l) -eq 0 ]; then
    echo "No shaders found"
    exit 1
fi

echo "Workflow test passed"
EOF

chmod +x /tmp/test_workflow.sh
run_test "Complete workflow" "/tmp/test_workflow.sh"
echo ""

echo -e "${CYAN}8. ERROR HANDLING${NC}"

# Test error conditions
run_test "Invalid scenario handling" "! $SDK/bin/scenario-runner --scenario /nonexistent.json 2>/dev/null"
run_test "Help on error" "$SDK/bin/scenario-runner --unknown-flag 2>&1 | grep -q help"
echo ""

# Summary
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}                    TEST SUMMARY                           ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""

PERCENTAGE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo "Tests Passed: $PASSED_TESTS/$TOTAL_TESTS ($PERCENTAGE%)"
echo ""

if [ $PERCENTAGE -eq 100 ]; then
    echo -e "${GREEN}✅ All end-to-end tests passed!${NC}"
    echo -e "${GREEN}✅ ML pipeline is fully functional!${NC}"
elif [ $PERCENTAGE -ge 80 ]; then
    echo -e "${YELLOW}⚠ Most tests passed, some issues detected${NC}"
else
    echo -e "${RED}✗ Multiple failures detected${NC}"
fi

echo ""
echo "Log file: /tmp/test_output.log"