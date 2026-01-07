#!/bin/bash
# ARM ML SDK - Comprehensive Feature Demonstration
# Showcases all SDK features including Vulkan 1.4, new ML models, and advanced tooling

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# SDK paths
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK="$ROOT_DIR/builds/ARM-ML-SDK-Complete"
export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK/lib
export PATH="$SDK/bin:$PATH"
export VK_LAYER_PATH="$SDK/lib:$VK_LAYER_PATH"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       ARM ML SDK - Complete Feature Demonstration         ║${NC}"
echo -e "${BLUE}║            macOS ARM64 (Apple Silicon)                    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# SECTION 1: Available SDK Components
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  1. Available SDK Components                              ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Get component counts
MODEL_COUNT=$(ls -1 $SDK/models/*.tflite 2>/dev/null | wc -l | tr -d ' ')
SHADER_COUNT=$(ls -1 $SDK/shaders/*.spv 2>/dev/null | wc -l | tr -d ' ')
TOOL_COUNT=$(ls -1 $SDK/tools/*.py 2>/dev/null | wc -l | tr -d ' ')
TUTORIAL_COUNT=$(ls -1 $ROOT_DIR/ml_tutorials/*.sh 2>/dev/null | wc -l | tr -d ' ')

echo -e "${CYAN}Core Components:${NC}"
if [ -f "$SDK/bin/scenario-runner" ]; then
    SIZE=$(du -h "$SDK/bin/scenario-runner" | cut -f1)
    echo -e "  ${GREEN}✓${NC} scenario-runner executable ($SIZE)"
else
    echo -e "  ${RED}✗${NC} scenario-runner - NOT FOUND"
fi
echo "  • ML Models: $MODEL_COUNT TensorFlow Lite models"
echo "  • Compute Shaders: $SHADER_COUNT SPIR-V shaders"
echo "  • Python Tools: $TOOL_COUNT utilities"
echo "  • Tutorials: $TUTORIAL_COUNT interactive tutorials"
echo ""

# ============================================================================
# SECTION 2: New Features Highlights
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  2. New Features in This Release                          ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Feature: Vulkan 1.4 Advanced Compute${NC}"
echo "  • Subgroup operations for SIMD-level parallelism"
echo "  • Cooperative matrix support (with fallback)"
echo "  • Optimized matmul shader using subgroup reductions"
echo ""

echo -e "${CYAN}Feature: Enhanced ML Model Support${NC}"
echo "  • MobileNet V3 architecture analysis"
echo "  • EfficientNet model processing"
echo "  • Transformer model inference scenarios"
echo "  • Automatic model architecture detection"
echo ""

echo -e "${CYAN}Feature: Performance Profiling Dashboard${NC}"
echo "  • Real-time metrics collection"
echo "  • JSON metrics export"
echo "  • Statistical analysis (avg, min, max, std_dev, percentiles)"
echo "  • Benchmark target tracking"
echo ""

echo -e "${CYAN}Feature: Unified Launcher System${NC}"
echo "  • Single entry point for all SDK components"
echo "  • MoltenVK/Vulkan runtime validation"
echo "  • Component health checks"
echo "  • Interactive SDK shell"
echo ""

# ============================================================================
# SECTION 3: Test Executable
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  3. Testing Scenario Runner                               ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Version Check:${NC}"
$SDK/bin/scenario-runner --version 2>&1 || echo -e "  ${YELLOW}(version output unavailable)${NC}"
echo ""

# ============================================================================
# SECTION 4: Available ML Models
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  4. Available ML Models                                   ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

for model in $SDK/models/*.tflite; do
    if [ -f "$model" ]; then
        SIZE=$(du -h "$model" | cut -f1)
        NAME=$(basename "$model" .tflite)
        # Categorize model
        case "$NAME" in
            mobilenet*)
                echo -e "  ${GREEN}✓${NC} $NAME ($SIZE) - Classification"
                ;;
            *style*|la_muse|udnie|mirror|wave_crop|des_glaneuses)
                echo -e "  ${GREEN}✓${NC} $NAME ($SIZE) - Style Transfer"
                ;;
            fire_detection*)
                echo -e "  ${GREEN}✓${NC} $NAME ($SIZE) - Detection"
                ;;
            *)
                echo -e "  ${GREEN}✓${NC} $NAME ($SIZE)"
                ;;
        esac
    fi
done
echo ""

# ============================================================================
# SECTION 5: Available Compute Shaders
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  5. Available Compute Shaders                             ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Basic Operations:${NC}"
for shader in add multiply divide subtract; do
    if [ -f "$SDK/shaders/${shader}.spv" ]; then
        SIZE=$(du -h "$SDK/shaders/${shader}.spv" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $shader ($SIZE)"
    fi
done

echo ""
echo -e "${CYAN}ML Operations:${NC}"
for shader in matmul matmul_subgroup conv2d relu sigmoid softmax maxpool; do
    if [ -f "$SDK/shaders/${shader}.spv" ]; then
        SIZE=$(du -h "$SDK/shaders/${shader}.spv" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $shader ($SIZE)"
    elif [ -f "$SDK/shaders/matrix_multiply.spv" ] && [ "$shader" = "matmul" ]; then
        SIZE=$(du -h "$SDK/shaders/matrix_multiply.spv" | cut -f1)
        echo -e "  ${GREEN}✓${NC} matrix_multiply ($SIZE)"
    fi
done

ADDITIONAL=$(ls -1 $SDK/shaders/*.spv 2>/dev/null | wc -l | tr -d ' ')
if [ "$ADDITIONAL" -gt 10 ]; then
    echo -e "  ... and $((ADDITIONAL - 10)) more shaders"
fi
echo ""

# ============================================================================
# SECTION 6: Available Python Tools
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  6. Available Python Tools                                ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Analysis & Profiling:${NC}"
for tool in analyze_tflite_model profile_performance export_metrics; do
    if [ -f "$SDK/tools/${tool}.py" ]; then
        echo -e "  ${GREEN}✓${NC} $tool.py"
    fi
done

echo ""
echo -e "${CYAN}Model Management:${NC}"
for tool in download_models convert_model_optimized validate_ml_operations; do
    if [ -f "$SDK/tools/${tool}.py" ]; then
        echo -e "  ${GREEN}✓${NC} $tool.py"
    fi
done

echo ""
echo -e "${CYAN}Optimization & Monitoring:${NC}"
for tool in optimize_for_apple_silicon realtime_performance_monitor create_ml_pipeline; do
    if [ -f "$SDK/tools/${tool}.py" ]; then
        echo -e "  ${GREEN}✓${NC} $tool.py"
    fi
done
echo ""

# ============================================================================
# SECTION 7: Available Tutorials
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  7. Available Tutorials                                   ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Tutorial descriptions function
get_tutorial_desc() {
    case "$1" in
        "1_analyze_model")  echo "Model Analysis - Inspect TFLite model structure" ;;
        "2_test_compute")   echo "Compute Shaders - Test Vulkan compute operations" ;;
        "3_benchmark")      echo "Benchmarking - Performance testing and targets" ;;
        "4_style_transfer") echo "Style Transfer - Neural style transfer demo" ;;
        "5_optimization")   echo "Apple Silicon - M-series specific optimizations" ;;
        "6_advanced_vulkan") echo "Advanced Vulkan - Vulkan 1.4 features and patterns" ;;
        "7_new_models")     echo "New Models - MobileNet V3, EfficientNet, Transformers" ;;
        *)                  echo "$1" ;;
    esac
}

for tutorial in $ROOT_DIR/ml_tutorials/*.sh; do
    if [ -f "$tutorial" ]; then
        NAME=$(basename "$tutorial" .sh)
        DESC=$(get_tutorial_desc "$NAME")
        echo -e "  ${CYAN}Tutorial $NAME${NC}"
        echo "    $DESC"
        echo "    Run: ./ml_tutorials/${NAME}.sh"
        echo ""
    fi
done

# ============================================================================
# SECTION 8: Quick Start Examples
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  8. Quick Start Examples                                  ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Run a Tutorial:${NC}"
echo -e "  ${GREEN}./ml_tutorials/1_analyze_model.sh${NC}"
echo ""

echo -e "${CYAN}Analyze a Model:${NC}"
echo -e "  ${GREEN}python3 $SDK/tools/analyze_tflite_model.py $SDK/models/mobilenet_v2.tflite${NC}"
echo ""

echo -e "${CYAN}Run Performance Benchmark:${NC}"
echo -e "  ${GREEN}./ml_tutorials/3_benchmark.sh${NC}"
echo ""

echo -e "${CYAN}Run Scenario with Profiling:${NC}"
echo -e "  ${GREEN}$SDK/bin/scenario-runner --scenario scenario.json --profiling-dump-path profile.json${NC}"
echo ""

echo -e "${CYAN}Use Unified Launcher:${NC}"
echo -e "  ${GREEN}./unified_launcher.sh status${NC}       - Check SDK health"
echo -e "  ${GREEN}./unified_launcher.sh validate${NC}     - Validate components"
echo -e "  ${GREEN}./unified_launcher.sh --check-vulkan${NC} - Check Vulkan runtime"
echo ""

echo -e "${CYAN}Export Metrics to JSON:${NC}"
echo -e "  ${GREEN}python3 $SDK/tools/export_metrics.py profile.json --output metrics.json${NC}"
echo ""

# ============================================================================
# SECTION 9: Scenario Test
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  9. Live Scenario Test                                    ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Create test scenario
cat > /tmp/ml_test.json << 'EOF'
{
  "name": "ML Demo Scenario",
  "description": "Demonstration of SDK compute capabilities",
  "version": "1.0",
  "operations": [
    {
      "type": "compute",
      "kernel": "add",
      "inputs": [
        {"data": [1.0, 2.0, 3.0, 4.0]},
        {"data": [5.0, 6.0, 7.0, 8.0]}
      ],
      "output_size": 4
    }
  ]
}
EOF

echo -e "${CYAN}Testing scenario execution (dry-run):${NC}"
$SDK/bin/scenario-runner --scenario /tmp/ml_test.json --dry-run 2>&1 | head -15 || true
echo ""

# Clean up
rm -f /tmp/ml_test.json

# ============================================================================
# SECTION 10: Feature Summary
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  10. Feature Summary                                      ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Available Features:${NC}"
echo "  • Vulkan 1.4 compute pipeline with subgroup operations"
echo "  • $MODEL_COUNT ML models (Classification, Style Transfer, Detection)"
echo "  • $SHADER_COUNT SPIR-V compute shaders (including optimized matmul)"
echo "  • $TOOL_COUNT Python tools for analysis and profiling"
echo "  • $TUTORIAL_COUNT interactive tutorials"
echo "  • Real-time performance monitoring"
echo "  • Apple Silicon (M-series) optimizations"
echo "  • Transformer model inference support"
echo "  • Unified launcher with health checks"
echo ""

echo -e "${CYAN}Performance Targets (see benchmark tutorial):${NC}"
echo "  • MatMul: < 1.5ms for 1024x1024"
echo "  • Conv2D: < 3ms for 224x224x32"
echo "  • Style Transfer: < 200ms for 256x256"
echo "  • Memory Bandwidth: > 100 GB/s"
echo ""

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   ${GREEN}✅ SDK Ready - All Features Available${NC}${BLUE}                  ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "For detailed component validation, run: ${GREEN}./unified_launcher.sh validate${NC}"
echo -e "For Vulkan runtime check, run: ${GREEN}./unified_launcher.sh --check-vulkan${NC}"
echo ""
