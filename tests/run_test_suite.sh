#!/bin/bash
# Master Test Runner for Vulkan ML SDK
# Comprehensive test suite with multiple levels and reporting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Paths
SDK_ROOT="/Users/jerry/Vulkan"
SDK_BUILD="$SDK_ROOT/builds/ARM-ML-SDK-Complete"
TEST_ROOT="$SDK_ROOT/tests"
REPORT_DIR="$TEST_ROOT/reports"

# Test configuration
TEST_LEVEL="${1:-standard}"  # quick, standard, or extensive
PARALLEL="${2:-true}"
MAX_WORKERS="${3:-4}"

# Export SDK path for tests
export SDK_PATH="$SDK_BUILD"
export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK_BUILD/lib

# Statistics
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0
START_TIME=$(date +%s)

# Create report directory
mkdir -p "$REPORT_DIR"
REPORT_FILE="$REPORT_DIR/test_report_$(date +%Y%m%d_%H%M%S).json"
LOG_FILE="$REPORT_DIR/test_log_$(date +%Y%m%d_%H%M%S).log"

# Function to print header
print_header() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          Vulkan ML SDK - Comprehensive Test Suite         ║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║  Level: $(echo $TEST_LEVEL | tr '[:lower:]' '[:upper:]')                                         ║${NC}"
    echo -e "${BLUE}║  Platform: macOS ARM64 (Apple M4 Max)                     ║${NC}"
    echo -e "${BLUE}║  SDK: ARM-ML-SDK-Complete                                 ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Function to run a test category
run_test_category() {
    local category=$1
    local description=$2
    
    echo -e "${CYAN}═══ $description ═══${NC}"
    echo ""
    
    local category_passed=0
    local category_failed=0
    
    # Find tests in category
    if [ -d "$TEST_ROOT/$category" ]; then
        # Python tests
        for test_file in $(find "$TEST_ROOT/$category" -name "test_*.py" 2>/dev/null); do
            run_single_test "$test_file" "Python"
            if [ $? -eq 0 ]; then
                category_passed=$((category_passed + 1))
            else
                category_failed=$((category_failed + 1))
            fi
        done
        
        # C++ tests
        for test_file in $(find "$TEST_ROOT/$category" -name "test_*.cpp" 2>/dev/null); do
            # Compile and run C++ test
            compile_and_run_cpp "$test_file"
            if [ $? -eq 0 ]; then
                category_passed=$((category_passed + 1))
            else
                category_failed=$((category_failed + 1))
            fi
        done
        
        # Shell tests
        for test_file in $(find "$TEST_ROOT/$category" -name "test_*.sh" 2>/dev/null); do
            run_single_test "$test_file" "Shell"
            if [ $? -eq 0 ]; then
                category_passed=$((category_passed + 1))
            else
                category_failed=$((category_failed + 1))
            fi
        done
    fi
    
    echo ""
    echo -e "Category Results: ${GREEN}$category_passed passed${NC}, ${RED}$category_failed failed${NC}"
    echo ""
    
    PASSED_TESTS=$((PASSED_TESTS + category_passed))
    FAILED_TESTS=$((FAILED_TESTS + category_failed))
}

# Function to run a single test
run_single_test() {
    local test_file=$1
    local test_type=$2
    local test_name=$(basename "$test_file")
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -ne "  Running $test_name... "
    
    local start=$(date +%s%N)
    
    # Run test based on type
    if [ "$test_type" = "Python" ]; then
        if python3 "$test_file" >> "$LOG_FILE" 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            return 0
        else
            echo -e "${RED}FAIL${NC}"
            return 1
        fi
    elif [ "$test_type" = "Shell" ]; then
        if bash "$test_file" >> "$LOG_FILE" 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            return 0
        else
            echo -e "${RED}FAIL${NC}"
            return 1
        fi
    fi
    
    local end=$(date +%s%N)
    local duration=$(( (end - start) / 1000000 ))
    echo " (${duration}ms)"
}

# Function to compile and run C++ test
compile_and_run_cpp() {
    local cpp_file=$1
    local test_name=$(basename "$cpp_file" .cpp)
    local executable="$TEST_ROOT/bin/$test_name"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -ne "  Compiling and running $test_name... "
    
    # Create bin directory if it doesn't exist
    mkdir -p "$TEST_ROOT/bin"
    
    # Compile with all necessary includes and libraries
    if c++ -std=c++17 -O2 -march=native \
           -I"$SDK_BUILD/include" \
           -I"/Users/jerry/Vulkan/ai-ml-sdk-vgf-library/include" \
           -I"/usr/local/include" \
           -I"/opt/homebrew/include" \
           -L"$SDK_BUILD/lib" \
           -L"/usr/local/lib" \
           -L"/opt/homebrew/lib" \
           "$cpp_file" -o "$executable" \
           -lvgf -lvulkan -lpthread -framework Metal -framework Foundation 2>> "$LOG_FILE"; then
        
        # Run
        if "$executable" >> "$LOG_FILE" 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            return 0
        else
            echo -e "${RED}FAIL${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}COMPILE ERROR${NC}"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        return 2
    fi
}

# Function to run quick tests (5 minutes)
run_quick_tests() {
    echo -e "${MAGENTA}Running QUICK Test Suite (5 minutes)${NC}"
    echo ""
    
    # Basic smoke tests
    run_test_category "unit/vgf" "VGF Library Unit Tests"
    run_test_category "validation/numerical" "Numerical Validation"
    
    # Quick performance check
    echo -e "${CYAN}═══ Quick Performance Check ═══${NC}"
    python3 -c "
import numpy as np
import time

# Memory bandwidth test
size = 1_000_000
data = np.arange(size, dtype=np.float32)
start = time.time()
for _ in range(100):
    result = data * 2.0
elapsed = time.time() - start
bandwidth = (size * 4 * 100 * 2) / elapsed / (1024**3)
print(f'  Memory Bandwidth: {bandwidth:.2f} GB/s')

# Matrix multiply test
N = 256
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
start = time.time()
C = np.matmul(A, B)
elapsed = (time.time() - start) * 1000
gflops = (2 * N**3) / (elapsed / 1000) / 1e9
print(f'  MatMul Performance: {gflops:.2f} GFLOPS')
"
    echo ""
}

# Function to run standard tests (30 minutes)
run_standard_tests() {
    echo -e "${MAGENTA}Running STANDARD Test Suite (30 minutes)${NC}"
    echo ""
    
    # Unit tests
    run_test_category "unit/vgf" "VGF Library Unit Tests"
    run_test_category "unit/converter" "Model Converter Tests"
    run_test_category "unit/runner" "Scenario Runner Tests"
    run_test_category "unit/emulation" "Emulation Layer Tests"
    
    # Integration tests
    run_test_category "integration/model_pipeline" "Model Pipeline Tests"
    run_test_category "integration/inference" "Inference Tests"
    run_test_category "integration/shader_execution" "Shader Execution Tests"
    
    # Performance tests
    run_test_category "performance/operations" "Operation Benchmarks"
    run_test_category "performance/models" "Model Benchmarks"
    run_test_category "performance/memory" "Memory Tests"
    
    # Validation tests
    run_test_category "validation/numerical" "Numerical Validation"
    run_test_category "validation/ml_operations" "ML Operation Validation"
    run_test_category "validation/outputs" "Output Validation"
}

# Function to run extensive tests (2+ hours)
run_extensive_tests() {
    echo -e "${MAGENTA}Running EXTENSIVE Test Suite (2+ hours)${NC}"
    echo ""
    
    # All standard tests
    run_standard_tests
    
    # Stress tests
    run_test_category "stress/large_models" "Large Model Tests"
    run_test_category "stress/concurrent" "Concurrent Execution Tests"
    run_test_category "stress/resource_limits" "Resource Limit Tests"
    
    # Regression tests
    run_test_category "regression/known_issues" "Known Issue Tests"
    run_test_category "regression/compatibility" "Compatibility Tests"
    
    # Platform-specific tests
    run_test_category "platform/macos_arm64" "macOS ARM64 Tests"
}

# Function to generate test scenarios if needed
generate_test_scenarios() {
    echo -e "${CYAN}Generating test scenarios...${NC}"
    
    if [ -f "$TEST_ROOT/framework/test_scenarios.py" ]; then
        python3 "$TEST_ROOT/framework/test_scenarios.py" >> "$LOG_FILE" 2>&1
        echo "  Test scenarios generated"
    else
        echo "  Scenario generator not found, skipping"
    fi
    echo ""
}

# Function to run Python test framework
run_python_framework() {
    echo -e "${CYAN}Running Python Test Framework...${NC}"
    
    if [ -f "$TEST_ROOT/framework/test_framework.py" ]; then
        python3 "$TEST_ROOT/framework/test_framework.py" \
                --level "$TEST_LEVEL" \
                --parallel \
                --workers "$MAX_WORKERS" \
                --report "$REPORT_FILE" >> "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}Framework tests completed${NC}"
        else
            echo -e "  ${RED}Framework tests failed${NC}"
        fi
    else
        echo "  Test framework not found, skipping"
    fi
    echo ""
}

# Function to generate HTML report
generate_html_report() {
    local html_file="$REPORT_DIR/test_report_$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$html_file" << 'HTML'
<!DOCTYPE html>
<html>
<head>
    <title>Vulkan ML SDK Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .summary { background: #f0f0f0; padding: 15px; border-radius: 5px; }
        .passed { color: green; font-weight: bold; }
        .failed { color: red; font-weight: bold; }
        .skipped { color: orange; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .chart { width: 100%; height: 300px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Vulkan ML SDK Test Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Test results will be displayed here</p>
    </div>
</body>
</html>
HTML
    
    echo "  HTML report generated: $html_file"
}

# Function to print final summary
print_summary() {
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    local MINUTES=$((DURATION / 60))
    local SECONDS=$((DURATION % 60))
    
    echo ""
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}                    TEST SUITE SUMMARY                      ${NC}"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    echo "Execution Time: ${MINUTES}m ${SECONDS}s"
    echo "Total Tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo -e "Skipped: ${YELLOW}$SKIPPED_TESTS${NC}"
    
    if [ $TOTAL_TESTS -gt 0 ]; then
        local SUCCESS_RATE=$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)
        echo "Success Rate: ${SUCCESS_RATE}%"
        
        if (( $(echo "$SUCCESS_RATE >= 95" | bc -l) )); then
            echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${GREEN}║         ✅ EXCELLENT - SDK IS PRODUCTION READY!           ║${NC}"
            echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        elif (( $(echo "$SUCCESS_RATE >= 80" | bc -l) )); then
            echo -e "\n${GREEN}✅ GOOD - Minor issues detected${NC}"
        elif (( $(echo "$SUCCESS_RATE >= 60" | bc -l) )); then
            echo -e "\n${YELLOW}⚠️  FAIR - Needs improvement${NC}"
        else
            echo -e "\n${RED}❌ POOR - Major issues detected${NC}"
        fi
    fi
    
    echo ""
    echo "Reports:"
    echo "  JSON: $REPORT_FILE"
    echo "  Log: $LOG_FILE"
    
    # Generate HTML report
    generate_html_report
}

# Main execution
main() {
    print_header
    
    # Check dependencies
    echo -e "${CYAN}Checking dependencies...${NC}"
    command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
    command -v c++ >/dev/null 2>&1 || { echo "C++ compiler required"; exit 1; }
    python3 -c "import numpy" 2>/dev/null || { echo "NumPy required"; exit 1; }
    echo "  All dependencies satisfied"
    echo ""
    
    # Generate test scenarios if needed
    generate_test_scenarios
    
    # Run tests based on level
    case $TEST_LEVEL in
        quick)
            run_quick_tests
            ;;
        standard)
            run_standard_tests
            ;;
        extensive)
            run_extensive_tests
            ;;
        *)
            echo "Invalid test level: $TEST_LEVEL"
            echo "Use: quick, standard, or extensive"
            exit 1
            ;;
    esac
    
    # Run Python test framework
    run_python_framework
    
    # Generate reports and summary
    print_summary
}

# Handle interrupts
trap 'echo -e "\n${YELLOW}Tests interrupted${NC}"; print_summary; exit 1' INT TERM

# Run main function
main

exit $FAILED_TESTS