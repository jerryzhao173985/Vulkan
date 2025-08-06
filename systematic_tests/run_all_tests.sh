#!/bin/bash
# Master Test Runner - Systematic Testing Suite

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BLUE='\033[0;34m'
NC='\033[0m'

clear

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      ARM ML SDK - SYSTEMATIC TEST SUITE                   ║${NC}"
echo -e "${BLUE}║         Unit | Performance | End-to-End                   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Results tracking
TOTAL_SUITES=0
PASSED_SUITES=0
TEST_RESULTS=""

# Function to run test suite
run_suite() {
    local suite_name="$1"
    local suite_cmd="$2"
    
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    echo -e "${CYAN}═══ $suite_name ═══${NC}"
    echo ""
    
    if eval "$suite_cmd"; then
        echo -e "${GREEN}✓ $suite_name passed${NC}"
        PASSED_SUITES=$((PASSED_SUITES + 1))
        TEST_RESULTS="${TEST_RESULTS}✓ $suite_name\n"
    else
        echo -e "${RED}✗ $suite_name failed${NC}"
        TEST_RESULTS="${TEST_RESULTS}✗ $suite_name\n"
    fi
    echo ""
    echo "Press Enter to continue..."
    read
    clear
}

# Build tests if needed
if [ ! -f "unit_tests" ] || [ ! -f "performance_benchmarks" ]; then
    echo -e "${CYAN}Building test suites...${NC}"
    ./build_tests.sh
    echo ""
fi

# 1. Unit Tests
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}                    PHASE 1: UNIT TESTS                    ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Testing individual components and functions..."
echo ""

if [ -f "unit_tests" ]; then
    ./unit_tests
    echo ""
    echo -e "${GREEN}✓ Unit tests complete${NC}"
    PASSED_SUITES=$((PASSED_SUITES + 1))
else
    echo -e "${YELLOW}! Unit tests not available${NC}"
fi
TOTAL_SUITES=$((TOTAL_SUITES + 1))

echo ""
echo "Press Enter to continue to performance tests..."
read
clear

# 2. Performance Benchmarks
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}               PHASE 2: PERFORMANCE BENCHMARKS             ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Measuring performance of key operations..."
echo ""

if [ -f "performance_benchmarks" ]; then
    ./performance_benchmarks
    echo ""
    echo -e "${GREEN}✓ Performance benchmarks complete${NC}"
    PASSED_SUITES=$((PASSED_SUITES + 1))
else
    echo -e "${YELLOW}! Performance benchmarks not available${NC}"
fi
TOTAL_SUITES=$((TOTAL_SUITES + 1))

echo ""
echo "Press Enter to continue to end-to-end tests..."
read
clear

# 3. End-to-End Tests
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}              PHASE 3: END-TO-END PIPELINE                 ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Testing complete ML pipeline integration..."
echo ""

if [ -f "end_to_end_tests.sh" ]; then
    chmod +x end_to_end_tests.sh
    ./end_to_end_tests.sh
    echo ""
    echo -e "${GREEN}✓ End-to-end tests complete${NC}"
    PASSED_SUITES=$((PASSED_SUITES + 1))
else
    echo -e "${YELLOW}! End-to-end tests not available${NC}"
fi
TOTAL_SUITES=$((TOTAL_SUITES + 1))

echo ""
echo "Press Enter to see final report..."
read
clear

# 4. Final Report
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    FINAL TEST REPORT                      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# System Information
echo -e "${CYAN}SYSTEM INFORMATION${NC}"
echo "Platform: macOS ARM64"
echo "Processor: $(sysctl -n machdep.cpu.brand_string)"
echo "Memory: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')"
echo "Cores: $(sysctl -n hw.ncpu)"
echo ""

# Test Results Summary
echo -e "${CYAN}TEST RESULTS${NC}"
echo "Test Suites Passed: $PASSED_SUITES/$TOTAL_SUITES"
echo ""

# Performance Highlights (if available)
if [ -f "/tmp/performance_summary.txt" ]; then
    echo -e "${CYAN}PERFORMANCE HIGHLIGHTS${NC}"
    cat /tmp/performance_summary.txt
    echo ""
fi

# Overall Status
PERCENTAGE=$((PASSED_SUITES * 100 / TOTAL_SUITES))
echo -e "${CYAN}OVERALL STATUS${NC}"

if [ $PERCENTAGE -eq 100 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║        ✅ ALL SYSTEMATIC TESTS PASSED!                    ║${NC}"
    echo -e "${GREEN}║        ✅ SDK IS PRODUCTION READY!                        ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
elif [ $PERCENTAGE -ge 66 ]; then
    echo -e "${YELLOW}⚠ Most tests passed ($PERCENTAGE%)${NC}"
else
    echo -e "${RED}✗ Multiple test failures ($PERCENTAGE% passed)${NC}"
fi

echo ""
echo "Test artifacts:"
echo "  • Unit test results: ./unit_tests"
echo "  • Performance results: ./performance_benchmarks"
echo "  • Integration logs: /tmp/test_output.log"
echo ""
echo -e "${CYAN}Test suite complete!${NC}"