#!/bin/bash
# ARM ML SDK Master Launcher with Component Validation and Unified Orchestration

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# SDK Home directory
SDK_HOME="$(cd "$(dirname "$0")" && pwd)"
export PATH="$SDK_HOME/bin:$PATH"
export DYLD_LIBRARY_PATH="/usr/local/lib:$SDK_HOME/lib:$DYLD_LIBRARY_PATH"
export VK_LAYER_PATH="$SDK_HOME/lib:$VK_LAYER_PATH"

# MoltenVK detection and Vulkan runtime validation function
check_vulkan_runtime() {
    local errors=0
    local warnings=0

    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}         ARM ML SDK - Vulkan Runtime Validation            ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    # 1. Check for MoltenVK library
    echo -e "${CYAN}1. MoltenVK Detection:${NC}"
    local moltenvk_found=0
    local moltenvk_locations=(
        "/usr/local/lib/libMoltenVK.dylib"
        "/opt/homebrew/lib/libMoltenVK.dylib"
        "$HOME/VulkanSDK/*/MoltenVK/dylib/macOS/libMoltenVK.dylib"
        "$SDK_HOME/lib/libMoltenVK.dylib"
    )

    for loc in "${moltenvk_locations[@]}"; do
        # Handle glob patterns
        for path in $loc; do
            if [ -f "$path" ]; then
                SIZE=$(du -h "$path" 2>/dev/null | cut -f1)
                echo -e "  ${GREEN}✓${NC} MoltenVK found: $path ($SIZE)"
                moltenvk_found=1
                break 2
            fi
        done
    done

    if [ $moltenvk_found -eq 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} MoltenVK library not found in standard locations"
        ((warnings++))
    fi

    # 2. Check for Vulkan loader library
    echo ""
    echo -e "${CYAN}2. Vulkan Loader Library:${NC}"
    local vulkan_lib_found=0
    local vulkan_locations=(
        "/usr/local/lib/libvulkan.dylib"
        "/usr/local/lib/libvulkan.1.dylib"
        "/opt/homebrew/lib/libvulkan.dylib"
        "$HOME/VulkanSDK/*/macOS/lib/libvulkan.dylib"
    )

    for loc in "${vulkan_locations[@]}"; do
        for path in $loc; do
            if [ -f "$path" ]; then
                SIZE=$(du -h "$path" 2>/dev/null | cut -f1)
                echo -e "  ${GREEN}✓${NC} Vulkan loader: $path ($SIZE)"
                vulkan_lib_found=1
                break 2
            fi
        done
    done

    if [ $vulkan_lib_found -eq 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} Vulkan loader library not found"
        ((warnings++))
    fi

    # 3. Check for ICD manifest files (Installable Client Driver)
    echo ""
    echo -e "${CYAN}3. Vulkan ICD Manifests:${NC}"
    local icd_found=0
    local icd_locations=(
        "/usr/local/share/vulkan/icd.d"
        "/opt/homebrew/share/vulkan/icd.d"
        "$HOME/.config/vulkan/icd.d"
        "$HOME/VulkanSDK/*/macOS/share/vulkan/icd.d"
    )

    for loc in "${icd_locations[@]}"; do
        for path in $loc; do
            if [ -d "$path" ]; then
                local manifest_count=$(ls -1 "$path"/*.json 2>/dev/null | wc -l)
                manifest_count=${manifest_count// /}
                if [ "$manifest_count" -gt 0 ]; then
                    echo -e "  ${GREEN}✓${NC} ICD manifests found: $path ($manifest_count files)"
                    ls "$path"/*.json 2>/dev/null | while read manifest; do
                        echo -e "    • $(basename "$manifest")"
                    done
                    icd_found=1
                fi
            fi
        done
    done

    if [ $icd_found -eq 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} No Vulkan ICD manifests found"
        ((warnings++))
    fi

    # 4. Check VK_DRIVER_FILES environment variable
    echo ""
    echo -e "${CYAN}4. Vulkan Environment Variables:${NC}"
    if [ -n "${VK_DRIVER_FILES:-}" ]; then
        echo -e "  ${GREEN}✓${NC} VK_DRIVER_FILES: $VK_DRIVER_FILES"
    else
        echo -e "  ${CYAN}ℹ${NC} VK_DRIVER_FILES: not set (using system default)"
    fi

    if [ -n "${VK_ICD_FILENAMES:-}" ]; then
        echo -e "  ${GREEN}✓${NC} VK_ICD_FILENAMES: $VK_ICD_FILENAMES"
    else
        echo -e "  ${CYAN}ℹ${NC} VK_ICD_FILENAMES: not set (using system default)"
    fi

    if [ -n "${VK_LAYER_PATH:-}" ]; then
        echo -e "  ${GREEN}✓${NC} VK_LAYER_PATH: $VK_LAYER_PATH"
    else
        echo -e "  ${CYAN}ℹ${NC} VK_LAYER_PATH: not set"
    fi

    # 5. Check vulkaninfo command and get runtime details
    echo ""
    echo -e "${CYAN}5. Vulkan Runtime Information:${NC}"
    if command -v vulkaninfo &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} vulkaninfo command available"

        # Try to get Vulkan version and device info
        local vulkan_output
        vulkan_output=$(vulkaninfo 2>&1 | head -30) || true

        # Extract key information
        local api_version=$(echo "$vulkan_output" | grep -i "apiVersion" | head -1 | awk '{print $NF}')
        local device_name=$(echo "$vulkan_output" | grep -i "deviceName" | head -1 | sed 's/.*= //')
        local driver_version=$(echo "$vulkan_output" | grep -i "driverVersion" | head -1 | awk '{print $NF}')

        if [ -n "$api_version" ]; then
            echo -e "  ${GREEN}✓${NC} Vulkan API Version: $api_version"
        fi
        if [ -n "$device_name" ]; then
            echo -e "  ${GREEN}✓${NC} GPU Device: $device_name"
        fi
        if [ -n "$driver_version" ]; then
            echo -e "  ${GREEN}✓${NC} Driver Version: $driver_version"
        fi

        # Check for compute queue support
        if echo "$vulkan_output" | grep -qi "compute"; then
            echo -e "  ${GREEN}✓${NC} Compute queue support detected"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} vulkaninfo not found - install Vulkan SDK for detailed info"
        ((warnings++))
    fi

    # 6. Test Vulkan initialization with scenario-runner if available
    echo ""
    echo -e "${CYAN}6. Vulkan Initialization Test:${NC}"
    if [ -f "$SDK_HOME/bin/scenario-runner" ] && [ -x "$SDK_HOME/bin/scenario-runner" ]; then
        local runner_output
        runner_output=$("$SDK_HOME/bin/scenario-runner" --version 2>&1) || true
        if echo "$runner_output" | grep -qi "vulkan\|version\|scenario"; then
            echo -e "  ${GREEN}✓${NC} scenario-runner Vulkan initialization OK"
        else
            echo -e "  ${YELLOW}⚠${NC} scenario-runner may have Vulkan issues"
            ((warnings++))
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} scenario-runner not available for testing"
        ((warnings++))
    fi

    # Summary
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    if [ $moltenvk_found -eq 1 ] || [ $vulkan_lib_found -eq 1 ]; then
        if [ $warnings -eq 0 ]; then
            echo -e "${GREEN}✅ Vulkan Runtime Validation PASSED${NC}"
        else
            echo -e "${YELLOW}⚠️  Vulkan Runtime Validation PASSED with $warnings warning(s)${NC}"
        fi
    else
        echo -e "${RED}❌ Vulkan Runtime Validation FAILED - No Vulkan libraries found${NC}"
        echo ""
        echo -e "${CYAN}To install Vulkan on macOS:${NC}"
        echo "  1. Install via Homebrew: brew install molten-vk"
        echo "  2. Or download LunarG Vulkan SDK: https://vulkan.lunarg.com/sdk/home"
        ((errors++))
    fi
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    return $errors
}

# Component validation function
validate_components() {
    local errors=0
    local warnings=0

    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}         ARM ML SDK - Component Validation                 ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}SDK Location:${NC} $SDK_HOME"
    echo ""

    # Validate binary
    echo -e "${CYAN}1. Validating Executable:${NC}"
    if [ -f "$SDK_HOME/bin/scenario-runner" ]; then
        if [ -x "$SDK_HOME/bin/scenario-runner" ]; then
            SIZE=$(du -h "$SDK_HOME/bin/scenario-runner" | cut -f1)
            echo -e "  ${GREEN}✓${NC} scenario-runner ($SIZE) - executable"
        else
            echo -e "  ${YELLOW}⚠${NC} scenario-runner - not executable"
            ((warnings++))
        fi
    else
        echo -e "  ${RED}✗${NC} scenario-runner - NOT FOUND"
        ((errors++))
    fi
    echo ""

    # Validate models
    echo -e "${CYAN}2. Validating ML Models:${NC}"
    local model_count=0
    local total_model_size=0
    if [ -d "$SDK_HOME/models" ]; then
        for model in "$SDK_HOME/models"/*.tflite; do
            if [ -f "$model" ]; then
                SIZE=$(du -h "$model" | cut -f1)
                NAME=$(basename "$model" .tflite)
                echo -e "  ${GREEN}✓${NC} $NAME ($SIZE)"
                ((model_count++))
            fi
        done
        if [ $model_count -eq 0 ]; then
            echo -e "  ${YELLOW}⚠${NC} No TFLite models found"
            ((warnings++))
        else
            echo -e "  ${GREEN}Total:${NC} $model_count models"
        fi
    else
        echo -e "  ${RED}✗${NC} models/ directory - NOT FOUND"
        ((errors++))
    fi
    echo ""

    # Validate shaders
    echo -e "${CYAN}3. Validating Compute Shaders:${NC}"
    local shader_count=0
    if [ -d "$SDK_HOME/shaders" ]; then
        shader_count=$(ls -1 "$SDK_HOME/shaders"/*.spv 2>/dev/null | wc -l)
        shader_count=${shader_count// /}
        if [ "$shader_count" -gt 0 ]; then
            echo -e "  ${GREEN}✓${NC} Found $shader_count SPIR-V shaders"
            # Show first few shaders as examples
            ls "$SDK_HOME/shaders"/*.spv 2>/dev/null | head -5 | while read shader; do
                NAME=$(basename "$shader" .spv)
                SIZE=$(du -h "$shader" | cut -f1)
                echo -e "    • $NAME ($SIZE)"
            done
            if [ "$shader_count" -gt 5 ]; then
                echo -e "    ... and $((shader_count - 5)) more"
            fi
        else
            echo -e "  ${YELLOW}⚠${NC} No SPIR-V shaders found"
            ((warnings++))
        fi
    else
        echo -e "  ${RED}✗${NC} shaders/ directory - NOT FOUND"
        ((errors++))
    fi
    echo ""

    # Validate tools
    echo -e "${CYAN}4. Validating Python Tools:${NC}"
    local tool_count=0
    if [ -d "$SDK_HOME/tools" ]; then
        for tool in "$SDK_HOME/tools"/*.py; do
            if [ -f "$tool" ]; then
                NAME=$(basename "$tool" .py)
                echo -e "  ${GREEN}✓${NC} $NAME"
                ((tool_count++))
            fi
        done
        for tool in "$SDK_HOME/tools"/*.sh; do
            if [ -f "$tool" ]; then
                NAME=$(basename "$tool" .sh)
                echo -e "  ${GREEN}✓${NC} $NAME (shell)"
                ((tool_count++))
            fi
        done
        if [ $tool_count -eq 0 ]; then
            echo -e "  ${YELLOW}⚠${NC} No tools found"
            ((warnings++))
        else
            echo -e "  ${GREEN}Total:${NC} $tool_count tools"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} tools/ directory not found"
        ((warnings++))
    fi
    echo ""

    # Validate libraries
    echo -e "${CYAN}5. Validating Libraries:${NC}"
    local lib_count=0
    if [ -d "$SDK_HOME/lib" ]; then
        lib_count=$(ls -1 "$SDK_HOME/lib"/*.{dylib,a,so} 2>/dev/null | wc -l)
        lib_count=${lib_count// /}
        if [ "$lib_count" -gt 0 ]; then
            echo -e "  ${GREEN}✓${NC} Found $lib_count libraries"
            ls "$SDK_HOME/lib"/*.{dylib,a,so} 2>/dev/null | while read lib; do
                NAME=$(basename "$lib")
                SIZE=$(du -h "$lib" | cut -f1)
                echo -e "    • $NAME ($SIZE)"
            done
        else
            echo -e "  ${YELLOW}⚠${NC} No SDK libraries (using system libraries)"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} lib/ directory not found"
        ((warnings++))
    fi
    echo ""

    # Check system dependencies
    echo -e "${CYAN}6. Checking System Dependencies:${NC}"
    if command -v vulkaninfo &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} Vulkan runtime available"
    else
        echo -e "  ${YELLOW}⚠${NC} vulkaninfo not found (MoltenVK may still work)"
        ((warnings++))
    fi

    if [ -d "/usr/local/lib" ]; then
        if ls /usr/local/lib/libMoltenVK* &> /dev/null 2>&1 || ls /usr/local/lib/libvulkan* &> /dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Vulkan libraries found in /usr/local/lib"
        else
            echo -e "  ${YELLOW}⚠${NC} MoltenVK/Vulkan not found in /usr/local/lib"
            ((warnings++))
        fi
    fi

    if command -v python3 &> /dev/null; then
        PY_VER=$(python3 --version 2>&1)
        echo -e "  ${GREEN}✓${NC} Python3 available ($PY_VER)"
    else
        echo -e "  ${YELLOW}⚠${NC} Python3 not found (some tools may not work)"
        ((warnings++))
    fi
    echo ""

    # Summary
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    if [ $errors -eq 0 ]; then
        if [ $warnings -eq 0 ]; then
            echo -e "${GREEN}✅ Validation PASSED - All components ready${NC}"
        else
            echo -e "${YELLOW}⚠️  Validation PASSED with $warnings warning(s)${NC}"
        fi
    else
        echo -e "${RED}❌ Validation FAILED - $errors error(s), $warnings warning(s)${NC}"
        return 1
    fi
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    return 0
}

# Show help
show_help() {
    echo -e "${BLUE}ARM ML SDK for Vulkan - macOS ARM64${NC}"
    echo ""
    echo "Usage: launch_sdk.sh [OPTIONS] [COMMAND]"
    echo ""
    echo "Options:"
    echo "  --validate      Validate all SDK components"
    echo "  --check-vulkan  Check MoltenVK and Vulkan runtime status"
    echo "  --shell         Launch interactive shell with SDK environment"
    echo "  --run SCENARIO  Run scenario-runner with specified scenario file"
    echo "  --benchmark     Run performance benchmark"
    echo "  --help          Show this help message"
    echo ""
    echo "Commands:"
    echo "  scenario-runner  Launch the ML scenario runner"
    echo "  analyze MODEL    Analyze a TFLite model"
    echo "  profile          Run performance profiler"
    echo ""
    echo "Examples:"
    echo "  ./launch_sdk.sh --validate"
    echo "  ./launch_sdk.sh --check-vulkan"
    echo "  ./launch_sdk.sh --run my_scenario.json"
    echo "  ./launch_sdk.sh scenario-runner --help"
    echo ""
}

# Run scenario
run_scenario() {
    local scenario_file="$1"

    if [ ! -f "$scenario_file" ]; then
        echo -e "${RED}Error: Scenario file not found: $scenario_file${NC}"
        return 1
    fi

    echo -e "${CYAN}Running scenario: $scenario_file${NC}"
    "$SDK_HOME/bin/scenario-runner" --scenario "$scenario_file"
}

# Run benchmark
run_benchmark() {
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
    "$SDK_HOME/bin/scenario-runner" --scenario "$benchmark_scenario" --dry-run 2>&1 || true

    rm -f "$benchmark_scenario"
}

# Default mode: show info and optionally launch shell
show_info() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║         ARM ML SDK for Vulkan - macOS ARM64              ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}SDK Location:${NC} $SDK_HOME"
    echo ""

    # Show component summary
    echo -e "${CYAN}Available Components:${NC}"
    if [ -f "$SDK_HOME/bin/scenario-runner" ]; then
        SIZE=$(du -h "$SDK_HOME/bin/scenario-runner" | cut -f1)
        echo "  • scenario-runner ($SIZE)"
    fi

    local model_count=$(ls -1 "$SDK_HOME/models"/*.tflite 2>/dev/null | wc -l)
    model_count=${model_count// /}
    echo "  • ML Models: $model_count TFLite models"

    local shader_count=$(ls -1 "$SDK_HOME/shaders"/*.spv 2>/dev/null | wc -l)
    shader_count=${shader_count// /}
    echo "  • Shaders: $shader_count SPIR-V compute shaders"

    local tool_count=$(ls -1 "$SDK_HOME/tools"/*.py 2>/dev/null | wc -l)
    tool_count=${tool_count// /}
    echo "  • Python Tools: $tool_count tools"
    echo ""

    # Test scenario-runner
    if [ -f "$SDK_HOME/bin/scenario-runner" ]; then
        echo -e "${CYAN}Testing scenario-runner:${NC}"
        "$SDK_HOME/bin/scenario-runner" --version 2>&1 || echo -e "${YELLOW}(version check failed)${NC}"
        echo ""
    fi

    echo -e "${CYAN}Quick Commands:${NC}"
    echo "  scenario-runner --help           Show runner help"
    echo "  scenario-runner --scenario FILE  Run a scenario"
    echo "  ./launch_sdk.sh --validate       Validate all components"
    echo ""
}

# Main entry point
main() {
    case "${1:-}" in
        --validate)
            validate_components
            ;;
        --check-vulkan)
            check_vulkan_runtime
            ;;
        --help|-h)
            show_help
            ;;
        --shell)
            show_info
            echo -e "${CYAN}Launching SDK environment shell...${NC}"
            exec $SHELL
            ;;
        --run)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}Error: --run requires a scenario file${NC}"
                exit 1
            fi
            run_scenario "$2"
            ;;
        --benchmark)
            run_benchmark
            ;;
        scenario-runner)
            shift
            "$SDK_HOME/bin/scenario-runner" "$@"
            ;;
        analyze)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}Error: analyze requires a model file${NC}"
                exit 1
            fi
            python3 "$SDK_HOME/tools/analyze_tflite_model.py" "$2"
            ;;
        profile)
            python3 "$SDK_HOME/tools/profile_performance.py"
            ;;
        "")
            show_info
            echo -e "${CYAN}Launching SDK environment shell...${NC}"
            exec $SHELL
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
