#!/bin/bash
# ARM ML SDK - Unified Launcher
# Main entry point for all SDK components with health checks and unified orchestration
# shellcheck disable=SC2034  # Color variables are used via indirection

set -euo pipefail

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Root directory (where this script lives)
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK="$ROOT_DIR/builds/ARM-ML-SDK-Complete"
export PATH="$SDK/bin:$PATH"
export DYLD_LIBRARY_PATH="/usr/local/lib:$SDK/lib:${DYLD_LIBRARY_PATH:-}"
export VK_LAYER_PATH="$SDK/lib:${VK_LAYER_PATH:-}"

# Version
VERSION="1.0.0"

# Show help message
show_help() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       ARM ML SDK - Unified Launcher v${VERSION}              ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage: unified_launcher.sh [COMMAND] [OPTIONS]"
    echo ""
    echo -e "${CYAN}Commands:${NC}"
    echo "  status            Show SDK component status and health check"
    echo "  validate          Validate all SDK components (detailed)"
    echo "  demo              Run the ML demonstration"
    echo "  run SCENARIO      Run scenario-runner with specified scenario file"
    echo "  tutorial N        Run tutorial N (1-7)"
    echo "  benchmark         Run performance benchmark"
    echo "  analyze MODEL     Analyze a TFLite model"
    echo "  profile           Run performance profiler"
    echo "  shell             Launch interactive shell with SDK environment"
    echo ""
    echo -e "${CYAN}Options:${NC}"
    echo "  --help, -h        Show this help message"
    echo "  --version, -v     Show version information"
    echo "  --check-vulkan    Check Vulkan runtime availability"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  ./unified_launcher.sh status"
    echo "  ./unified_launcher.sh demo"
    echo "  ./unified_launcher.sh run scenario.json"
    echo "  ./unified_launcher.sh tutorial 1"
    echo "  ./unified_launcher.sh analyze models/mobilenet_v2.tflite"
    echo ""
    echo -e "${CYAN}Tutorials:${NC}"
    echo "  1: Analyze ML models"
    echo "  2: Test compute shaders"
    echo "  3: Run benchmarks"
    echo "  4: Style transfer demo"
    echo "  5: Apple Silicon optimizations"
    echo "  6: Advanced Vulkan features"
    echo "  7: New ML model architectures"
    echo ""
}

# Show version
show_version() {
    echo "ARM ML SDK Unified Launcher v${VERSION}"
    echo "SDK Location: $SDK"
    if [ -f "$SDK/bin/scenario-runner" ]; then
        echo -n "Scenario Runner: "
        "$SDK/bin/scenario-runner" --version 2>&1 || echo "(version unavailable)"
    fi
}

# Component health check (quick status)
health_check() {
    local status=0
    local checks_passed=0
    local checks_total=6

    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}         ARM ML SDK - Component Health Check               ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    # Check 1: SDK directory
    if [ -d "$SDK" ]; then
        echo -e "  ${GREEN}✓${NC} SDK directory found"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${RED}✗${NC} SDK directory NOT FOUND: $SDK"
        status=1
    fi

    # Check 2: Scenario runner binary
    if [ -f "$SDK/bin/scenario-runner" ] && [ -x "$SDK/bin/scenario-runner" ]; then
        SIZE=$(du -h "$SDK/bin/scenario-runner" | cut -f1)
        echo -e "  ${GREEN}✓${NC} scenario-runner executable ($SIZE)"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${RED}✗${NC} scenario-runner NOT FOUND or not executable"
        status=1
    fi

    # Check 3: ML Models
    local model_count=0
    if [ -d "$SDK/models" ]; then
        model_count=$(ls -1 "$SDK/models"/*.tflite 2>/dev/null | wc -l | tr -d ' ')
    fi
    if [ "$model_count" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} ML models available ($model_count TFLite models)"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${YELLOW}⚠${NC} No ML models found"
    fi

    # Check 4: Shaders
    local shader_count=0
    if [ -d "$SDK/shaders" ]; then
        shader_count=$(ls -1 "$SDK/shaders"/*.spv 2>/dev/null | wc -l | tr -d ' ')
    fi
    if [ "$shader_count" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} Compute shaders available ($shader_count SPIR-V shaders)"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${YELLOW}⚠${NC} No compute shaders found"
    fi

    # Check 5: Libraries
    local lib_count=0
    if [ -d "$SDK/lib" ]; then
        # Use || true to prevent pipefail from exiting when glob doesn't match
        lib_count=$(ls -1 "$SDK/lib"/*.dylib "$SDK/lib"/*.a "$SDK/lib"/*.so 2>/dev/null | wc -l | tr -d ' ' || true)
    fi
    if [ "$lib_count" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} SDK libraries available ($lib_count libraries)"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${YELLOW}⚠${NC} No SDK libraries (using system libraries)"
    fi

    # Check 6: Vulkan runtime
    if ls /usr/local/lib/libMoltenVK* >/dev/null 2>&1 || ls /usr/local/lib/libvulkan* >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Vulkan runtime available"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${YELLOW}⚠${NC} MoltenVK/Vulkan not found in /usr/local/lib"
    fi

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    if [ "$status" -eq 0 ]; then
        echo -e "${GREEN}✅ Health Check PASSED ($checks_passed/$checks_total checks)${NC}"
    else
        echo -e "${RED}❌ Health Check FAILED - Critical components missing${NC}"
    fi
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    return "$status"
}

# Detailed component validation
validate_components() {
    if [ -f "$SDK/launch_sdk.sh" ]; then
        "$SDK/launch_sdk.sh" --validate
    else
        # Fallback to inline validation
        local errors=0
        local warnings=0

        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}         ARM ML SDK - Component Validation                 ${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        echo -e "${CYAN}SDK Location:${NC} $SDK"
        echo ""

        # Validate binary
        echo -e "${CYAN}1. Validating Executable:${NC}"
        if [ -f "$SDK/bin/scenario-runner" ]; then
            if [ -x "$SDK/bin/scenario-runner" ]; then
                SIZE=$(du -h "$SDK/bin/scenario-runner" | cut -f1)
                echo -e "  ${GREEN}✓${NC} scenario-runner ($SIZE) - executable"
            else
                echo -e "  ${YELLOW}⚠${NC} scenario-runner - not executable"
                warnings=$((warnings + 1))
            fi
        else
            echo -e "  ${RED}✗${NC} scenario-runner - NOT FOUND"
            errors=$((errors + 1))
        fi
        echo ""

        # Validate models
        echo -e "${CYAN}2. Validating ML Models:${NC}"
        local model_count=0
        if [ -d "$SDK/models" ]; then
            for model in "$SDK/models"/*.tflite; do
                if [ -f "$model" ]; then
                    SIZE=$(du -h "$model" | cut -f1)
                    NAME=$(basename "$model" .tflite)
                    echo -e "  ${GREEN}✓${NC} $NAME ($SIZE)"
                    model_count=$((model_count + 1))
                fi
            done
            if [ "$model_count" -eq 0 ]; then
                echo -e "  ${YELLOW}⚠${NC} No TFLite models found"
                warnings=$((warnings + 1))
            else
                echo -e "  ${GREEN}Total:${NC} $model_count models"
            fi
        else
            echo -e "  ${RED}✗${NC} models/ directory - NOT FOUND"
            errors=$((errors + 1))
        fi
        echo ""

        # Validate shaders
        echo -e "${CYAN}3. Validating Compute Shaders:${NC}"
        local shader_count=0
        if [ -d "$SDK/shaders" ]; then
            shader_count=$(ls -1 "$SDK/shaders"/*.spv 2>/dev/null | wc -l | tr -d ' ')
            if [ "$shader_count" -gt 0 ]; then
                echo -e "  ${GREEN}✓${NC} Found $shader_count SPIR-V shaders"
                ls "$SDK/shaders"/*.spv 2>/dev/null | head -5 | while read -r shader; do
                    NAME=$(basename "$shader" .spv)
                    SIZE=$(du -h "$shader" | cut -f1)
                    echo -e "    - $NAME ($SIZE)"
                done
                if [ "$shader_count" -gt 5 ]; then
                    echo -e "    ... and $((shader_count - 5)) more"
                fi
            else
                echo -e "  ${YELLOW}⚠${NC} No SPIR-V shaders found"
                warnings=$((warnings + 1))
            fi
        else
            echo -e "  ${RED}✗${NC} shaders/ directory - NOT FOUND"
            errors=$((errors + 1))
        fi
        echo ""

        # Validate tools
        echo -e "${CYAN}4. Validating Python Tools:${NC}"
        local tool_count=0
        if [ -d "$SDK/tools" ]; then
            for tool in "$SDK/tools"/*.py; do
                if [ -f "$tool" ]; then
                    NAME=$(basename "$tool" .py)
                    echo -e "  ${GREEN}✓${NC} $NAME"
                    tool_count=$((tool_count + 1))
                fi
            done
            if [ "$tool_count" -eq 0 ]; then
                echo -e "  ${YELLOW}⚠${NC} No tools found"
                warnings=$((warnings + 1))
            else
                echo -e "  ${GREEN}Total:${NC} $tool_count tools"
            fi
        else
            echo -e "  ${YELLOW}⚠${NC} tools/ directory not found"
            warnings=$((warnings + 1))
        fi
        echo ""

        # Validate libraries
        echo -e "${CYAN}5. Validating Libraries:${NC}"
        local lib_count=0
        if [ -d "$SDK/lib" ]; then
            lib_count=$(ls -1 "$SDK/lib"/*.{dylib,a,so} 2>/dev/null | wc -l | tr -d ' ' || true)
            if [ "$lib_count" -gt 0 ]; then
                echo -e "  ${GREEN}✓${NC} Found $lib_count libraries"
                ls "$SDK/lib"/*.{dylib,a,so} 2>/dev/null | while read -r lib; do
                    NAME=$(basename "$lib")
                    SIZE=$(du -h "$lib" | cut -f1)
                    echo -e "    - $NAME ($SIZE)"
                done
            else
                echo -e "  ${YELLOW}⚠${NC} No SDK libraries (using system libraries)"
            fi
        else
            echo -e "  ${YELLOW}⚠${NC} lib/ directory not found"
            warnings=$((warnings + 1))
        fi
        echo ""

        # Check system dependencies
        echo -e "${CYAN}6. Checking System Dependencies:${NC}"
        if command -v vulkaninfo >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Vulkan runtime available (vulkaninfo found)"
        else
            echo -e "  ${YELLOW}⚠${NC} vulkaninfo not found (MoltenVK may still work)"
            warnings=$((warnings + 1))
        fi

        if ls /usr/local/lib/libMoltenVK* >/dev/null 2>&1 || ls /usr/local/lib/libvulkan* >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Vulkan libraries found in /usr/local/lib"
        else
            echo -e "  ${YELLOW}⚠${NC} MoltenVK/Vulkan not found in /usr/local/lib"
            warnings=$((warnings + 1))
        fi

        if command -v python3 >/dev/null 2>&1; then
            PY_VER=$(python3 --version 2>&1)
            echo -e "  ${GREEN}✓${NC} Python3 available ($PY_VER)"
        else
            echo -e "  ${YELLOW}⚠${NC} Python3 not found (some tools may not work)"
            warnings=$((warnings + 1))
        fi
        echo ""

        # Summary
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        if [ "$errors" -eq 0 ]; then
            if [ "$warnings" -eq 0 ]; then
                echo -e "${GREEN}✅ Validation PASSED - All components ready${NC}"
            else
                echo -e "${YELLOW}⚠️  Validation PASSED with $warnings warning(s)${NC}"
            fi
        else
            echo -e "${RED}❌ Validation FAILED - $errors error(s), $warnings warning(s)${NC}"
            return 1
        fi
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    fi
}

# Check Vulkan runtime
check_vulkan() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}         ARM ML SDK - Vulkan Runtime Check                 ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    # Check MoltenVK
    echo -e "${CYAN}Checking MoltenVK/Vulkan Libraries:${NC}"
    if ls /usr/local/lib/libMoltenVK* >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} MoltenVK found in /usr/local/lib"
        ls -la /usr/local/lib/libMoltenVK* 2>/dev/null | while read -r line; do
            echo "    $line"
        done
    else
        echo -e "  ${YELLOW}⚠${NC} MoltenVK not found in /usr/local/lib"
    fi
    echo ""

    if ls /usr/local/lib/libvulkan* >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Vulkan loader found in /usr/local/lib"
        ls -la /usr/local/lib/libvulkan* 2>/dev/null | while read -r line; do
            echo "    $line"
        done
    else
        echo -e "  ${YELLOW}⚠${NC} Vulkan loader not found in /usr/local/lib"
    fi
    echo ""

    # Check vulkaninfo
    echo -e "${CYAN}Checking Vulkan Tools:${NC}"
    if command -v vulkaninfo >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} vulkaninfo available"
        echo ""
        echo -e "${CYAN}Vulkan Device Info:${NC}"
        vulkaninfo --summary 2>&1 | head -20 || echo "  (unable to get device info)"
    else
        echo -e "  ${YELLOW}⚠${NC} vulkaninfo not found"
        echo ""
        echo "Install vulkaninfo via:"
        echo "  brew install vulkan-tools"
    fi
    echo ""

    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

# Run demo
run_demo() {
    if [ -f "$ROOT_DIR/run_ml_demo.sh" ]; then
        "$ROOT_DIR/run_ml_demo.sh"
    else
        echo -e "${RED}Error: Demo script not found: $ROOT_DIR/run_ml_demo.sh${NC}"
        exit 1
    fi
}

# Run tutorial
run_tutorial() {
    local tutorial_num="$1"
    local tutorial_script="$ROOT_DIR/ml_tutorials/${tutorial_num}_"

    # Find the tutorial script
    local found_script=""
    for script in "$ROOT_DIR/ml_tutorials/${tutorial_num}_"*.sh; do
        if [ -f "$script" ]; then
            found_script="$script"
            break
        fi
    done

    if [ -n "$found_script" ] && [ -f "$found_script" ]; then
        echo -e "${CYAN}Running tutorial $tutorial_num: $(basename "$found_script")${NC}"
        echo ""
        "$found_script"
    else
        echo -e "${RED}Error: Tutorial $tutorial_num not found${NC}"
        echo ""
        echo "Available tutorials:"
        for script in "$ROOT_DIR/ml_tutorials"/*.sh; do
            if [ -f "$script" ]; then
                echo "  $(basename "$script")"
            fi
        done
        exit 1
    fi
}

# Run scenario
run_scenario() {
    local scenario_file="$1"

    if [ ! -f "$scenario_file" ]; then
        echo -e "${RED}Error: Scenario file not found: $scenario_file${NC}"
        exit 1
    fi

    echo -e "${CYAN}Running scenario: $scenario_file${NC}"
    "$SDK/bin/scenario-runner" --scenario "$scenario_file"
}

# Run benchmark
run_benchmark() {
    if [ -f "$ROOT_DIR/ml_tutorials/3_benchmark.sh" ]; then
        "$ROOT_DIR/ml_tutorials/3_benchmark.sh"
    else
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}         ARM ML SDK - Performance Benchmark                ${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo ""

        # Create benchmark scenario
        local benchmark_scenario="/tmp/sdk_benchmark.json"
        cat > "$benchmark_scenario" << 'EOF'
{
  "name": "SDK Benchmark",
  "description": "Performance benchmark for ARM ML SDK",
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

        echo -e "${CYAN}Running benchmark scenario...${NC}"
        "$SDK/bin/scenario-runner" --scenario "$benchmark_scenario" --dry-run 2>&1 || true

        rm -f "$benchmark_scenario"
        echo ""
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    fi
}

# Analyze model
analyze_model() {
    local model_file="$1"

    if [ -f "$SDK/tools/analyze_tflite_model.py" ]; then
        python3 "$SDK/tools/analyze_tflite_model.py" "$model_file"
    else
        echo -e "${RED}Error: Model analyzer not found: $SDK/tools/analyze_tflite_model.py${NC}"
        exit 1
    fi
}

# Run profiler
run_profiler() {
    if [ -f "$SDK/tools/profile_performance.py" ]; then
        python3 "$SDK/tools/profile_performance.py" "$@"
    else
        echo -e "${RED}Error: Performance profiler not found: $SDK/tools/profile_performance.py${NC}"
        exit 1
    fi
}

# Launch interactive shell
launch_shell() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       ARM ML SDK - Interactive Shell                      ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}SDK Location:${NC} $SDK"
    echo -e "${CYAN}Environment configured with:${NC}"
    echo "  PATH includes: $SDK/bin"
    echo "  DYLD_LIBRARY_PATH includes: $SDK/lib"
    echo ""
    echo -e "${CYAN}Quick commands:${NC}"
    echo "  scenario-runner --help    Show runner options"
    echo "  scenario-runner --version Show version"
    echo "  exit                      Exit SDK shell"
    echo ""
    exec "${SHELL:-/bin/bash}"
}

# Main entry point
main() {
    case "${1:-}" in
        status)
            health_check
            ;;
        validate)
            validate_components
            ;;
        demo)
            run_demo
            ;;
        run)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}Error: 'run' requires a scenario file${NC}"
                echo "Usage: unified_launcher.sh run SCENARIO_FILE"
                exit 1
            fi
            run_scenario "$2"
            ;;
        tutorial)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}Error: 'tutorial' requires a tutorial number (1-7)${NC}"
                echo "Usage: unified_launcher.sh tutorial N"
                exit 1
            fi
            run_tutorial "$2"
            ;;
        benchmark)
            run_benchmark
            ;;
        analyze)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}Error: 'analyze' requires a model file${NC}"
                echo "Usage: unified_launcher.sh analyze MODEL_FILE"
                exit 1
            fi
            analyze_model "$2"
            ;;
        profile)
            shift
            run_profiler "$@"
            ;;
        shell)
            launch_shell
            ;;
        --check-vulkan)
            check_vulkan
            ;;
        --help|-h)
            show_help
            ;;
        --version|-v)
            show_version
            ;;
        "")
            # Default: show status and help
            health_check
            echo ""
            show_help
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
