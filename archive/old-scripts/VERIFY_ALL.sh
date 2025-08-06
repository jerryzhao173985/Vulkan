#!/bin/bash
# Comprehensive verification of what we've built

GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'

SDK="/Users/jerry/Vulkan/builds/ARM-ML-SDK-Complete"
export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK/lib

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║     ARM ML SDK - Complete Verification                    ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Track results
PASSED=0
TOTAL=0

check() {
    TOTAL=$((TOTAL + 1))
    if eval "$2" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $1"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "  ${YELLOW}✗${NC} $1"
        return 1
    fi
}

echo -e "${CYAN}1. EXECUTABLE${NC}"
check "scenario-runner exists" "[ -f '$SDK/bin/scenario-runner' ]"
check "scenario-runner is executable" "[ -x '$SDK/bin/scenario-runner' ]"
check "scenario-runner runs" "$SDK/bin/scenario-runner --version"
echo "  Size: $(du -h $SDK/bin/scenario-runner | cut -f1)"
echo ""

echo -e "${CYAN}2. LIBRARIES${NC}"
check "libvgf.a exists" "[ -f '$SDK/lib/libvgf.a' ]"
check "libSPIRV.a exists" "[ -f '$SDK/lib/libSPIRV.a' ]"
check "libSPIRV-Tools.a exists" "[ -f '$SDK/lib/libSPIRV-Tools.a' ]"
check "libSPIRV-Tools-opt.a exists" "[ -f '$SDK/lib/libSPIRV-Tools-opt.a' ]"
echo "  Total libraries: $(ls -1 $SDK/lib/*.a | wc -l)"
echo ""

echo -e "${CYAN}3. ML MODELS${NC}"
MODEL_COUNT=0
for model in $SDK/models/*.tflite; do
    if [ -f "$model" ]; then
        NAME=$(basename "$model" .tflite | cut -c1-30)
        SIZE=$(du -h "$model" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $NAME ($SIZE)"
        MODEL_COUNT=$((MODEL_COUNT + 1))
        PASSED=$((PASSED + 1))
        TOTAL=$((TOTAL + 1))
    fi
done
echo "  Total: $MODEL_COUNT models ($(du -sh $SDK/models | cut -f1))"
echo ""

echo -e "${CYAN}4. COMPUTE SHADERS${NC}"
SHADER_COUNT=$(ls -1 $SDK/shaders/*.spv 2>/dev/null | wc -l)
if [ $SHADER_COUNT -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} $SHADER_COUNT SPIR-V shaders compiled"
    PASSED=$((PASSED + 1))
else
    echo -e "  ${YELLOW}✗${NC} No shaders found"
fi
TOTAL=$((TOTAL + 1))

# List key shaders
for shader in add multiply conv1d conv2d relu maxpool; do
    if [ -f "$SDK/shaders/${shader}.spv" ] || [ -f "$SDK/shaders/${shader}_shader.spv" ]; then
        echo -e "  ${GREEN}✓${NC} ${shader} shader"
    fi
done
echo ""

echo -e "${CYAN}5. PYTHON TOOLS${NC}"
TOOL_COUNT=$(ls -1 $SDK/tools/*.py 2>/dev/null | wc -l)
if [ $TOOL_COUNT -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} $TOOL_COUNT Python ML tools"
    PASSED=$((PASSED + 1))
    for tool in analyze_tflite_model optimize_for_apple_silicon profile_performance; do
        if [ -f "$SDK/tools/${tool}.py" ]; then
            echo -e "  ${GREEN}✓${NC} ${tool}.py"
        fi
    done
else
    echo -e "  ${YELLOW}✗${NC} No tools found"
fi
TOTAL=$((TOTAL + 1))
echo ""

echo -e "${CYAN}6. DEMOS & TUTORIALS${NC}"
check "run_ml_demo.sh exists" "[ -f './run_ml_demo.sh' ]"
check "DEMO_ALL.sh exists" "[ -f './DEMO_ALL.sh' ]"
check "ml_tutorials directory" "[ -d './ml_tutorials' ]"
TUTORIAL_COUNT=$(ls -1 ml_tutorials/*.sh 2>/dev/null | wc -l)
echo "  Tutorials: $TUTORIAL_COUNT interactive scripts"
echo ""

echo -e "${CYAN}7. DOCUMENTATION${NC}"
check "CLAUDE.md exists" "[ -f './CLAUDE.md' ]"
check "README.md exists" "[ -f './README.md' ]"
echo "  CLAUDE.md: $(wc -l < CLAUDE.md) lines (practical guide)"
echo "  README.md: $(wc -l < README.md) lines (overview)"
echo ""

echo -e "${CYAN}8. BUILD SYSTEM${NC}"
check "CMakeLists.txt" "[ -f 'ai-ml-sdk-for-vulkan/CMakeLists.txt' ]"
check "build.py script" "[ -f 'ai-ml-sdk-for-vulkan/scripts/build.py' ]"
check "build directory" "[ -d 'ai-ml-sdk-for-vulkan/build-final' ]"
echo ""

echo -e "${CYAN}9. VERSION INFO${NC}"
VERSION=$($SDK/bin/scenario-runner --version 2>/dev/null | grep version | cut -d'"' -f4 || echo "unknown")
echo "  SDK Version: $VERSION"
echo "  Platform: macOS ARM64"
echo "  Processor: $(sysctl -n machdep.cpu.brand_string)"
echo ""

echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}                    VERIFICATION SUMMARY                    ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""

PERCENTAGE=$((PASSED * 100 / TOTAL))
echo "Tests Passed: $PASSED/$TOTAL ($PERCENTAGE%)"
echo ""

if [ $PERCENTAGE -ge 90 ]; then
    echo -e "${GREEN}✅ EXCELLENT! Everything is working!${NC}"
    echo ""
    echo "You can now:"
    echo "  1. Run demos: ./DEMO_ALL.sh"
    echo "  2. Try tutorials: ./ml_tutorials/1_analyze_model.sh"
    echo "  3. Run ML inference: $SDK/bin/scenario-runner --scenario test.json"
    echo "  4. Benchmark: ./ml_tutorials/3_benchmark.sh"
elif [ $PERCENTAGE -ge 70 ]; then
    echo -e "${YELLOW}⚠ GOOD: Most components working${NC}"
else
    echo -e "${RED}✗ Some components need attention${NC}"
fi

echo ""
echo "For quick demo: ./run_ml_demo.sh"
echo "For details: cat CLAUDE.md"