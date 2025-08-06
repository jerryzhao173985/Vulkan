#!/bin/bash
# Build C++ Test Suites

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${CYAN}Building C++ Test Suites...${NC}"
echo ""

# Compile unit tests
echo "1. Building unit tests..."
c++ -std=c++17 -O2 -march=native \
    unit_tests.cpp \
    -o unit_tests \
    -pthread

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Unit tests built${NC}"
else
    echo -e "${YELLOW}✗ Unit tests build failed${NC}"
fi

# Compile performance benchmarks
echo "2. Building performance benchmarks..."
c++ -std=c++17 -O3 -march=native \
    performance_benchmarks.cpp \
    -o performance_benchmarks \
    -pthread

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Performance benchmarks built${NC}"
else
    echo -e "${YELLOW}✗ Performance benchmarks build failed${NC}"
fi

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Available tests:"
echo "  ./unit_tests              - Run unit tests"
echo "  ./performance_benchmarks  - Run performance benchmarks"
echo "  ./end_to_end_tests.sh    - Run end-to-end tests"
echo "  ./run_all_tests.sh       - Run everything"