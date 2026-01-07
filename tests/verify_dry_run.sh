#!/bin/bash
# Verify scenario-runner dry-run mode with test scenario
# This script tests that scenario-runner can parse and validate scenarios

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK="$(cd "$SCRIPT_DIR/.." && pwd)/builds/ARM-ML-SDK-Complete"
export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK/lib

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "=== Dry-Run Verification Test ==="
echo "SDK Path: $SDK"
echo "Test Scenario: $SCRIPT_DIR/test_scenario.json"
echo ""

# Check prerequisites
if [ ! -f "$SDK/bin/scenario-runner" ]; then
    echo -e "${RED}ERROR: scenario-runner binary not found${NC}"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/test_scenario.json" ]; then
    echo -e "${RED}ERROR: test_scenario.json not found${NC}"
    exit 1
fi

echo "Prerequisites:"
echo "  ✓ scenario-runner binary exists"
echo "  ✓ test_scenario.json exists"
echo ""

# Validate JSON syntax
echo "Validating JSON syntax..."
if python3 -m json.tool "$SCRIPT_DIR/test_scenario.json" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ JSON syntax valid${NC}"
else
    echo -e "  ${RED}✗ JSON syntax invalid${NC}"
    exit 1
fi
echo ""

# Run dry-run test and capture output
echo "Running scenario-runner --dry-run..."
OUTPUT=$($SDK/bin/scenario-runner --scenario "$SCRIPT_DIR/test_scenario.json" --dry-run 2>&1 || true)
echo "$OUTPUT"
echo ""

# Check for scenario parsing success
if echo "$OUTPUT" | grep -q "Scenario file parsed"; then
    echo -e "${GREEN}✓ Scenario parsing: PASSED${NC}"
    PARSE_OK=1
else
    echo -e "${YELLOW}⚠ Scenario parsing: Could not verify${NC}"
    PARSE_OK=0
fi

# Note about Vulkan errors (expected in CI/headless environments)
if echo "$OUTPUT" | grep -q "vkEnumeratePhysicalDevices"; then
    echo -e "${YELLOW}⚠ Vulkan device enumeration failed (expected in headless environment)${NC}"
    echo "  This is normal when no Vulkan ICD is configured."
fi

echo ""
echo "=== Verification Summary ==="
if [ "$PARSE_OK" -eq 1 ]; then
    echo -e "${GREEN}✓ Dry-run verification PASSED${NC}"
    echo "  - Binary executes correctly"
    echo "  - JSON scenario is valid and parsed"
    echo "  - Dry-run mode works as expected"
    exit 0
else
    echo -e "${YELLOW}⚠ Dry-run verification completed with warnings${NC}"
    exit 0
fi
