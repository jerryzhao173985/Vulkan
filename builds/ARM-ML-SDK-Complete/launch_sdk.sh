#!/bin/bash
# ARM ML SDK Master Launcher

SDK_HOME="$(cd "$(dirname "$0")" && pwd)"
export PATH="$SDK_HOME/bin:$PATH"
export DYLD_LIBRARY_PATH="/usr/local/lib:$SDK_HOME/lib:$DYLD_LIBRARY_PATH"
export VK_LAYER_PATH="$SDK_HOME/lib:$VK_LAYER_PATH"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         ARM ML SDK for Vulkan - macOS ARM64              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "SDK Location: $SDK_HOME"
echo ""

# Show available tools
echo "Available Tools:"
echo "  • scenario-runner - Run Vulkan compute scenarios"
if [ -d "$SDK_HOME/tools" ]; then
    echo "  • ML Pipeline tools in tools/"
fi
if [ -d "$SDK_HOME/models" ]; then
    echo "  • ML Models in models/"
fi
echo ""

# Test scenario-runner
if [ -f "$SDK_HOME/bin/scenario-runner" ]; then
    echo "Testing scenario-runner..."
    "$SDK_HOME/bin/scenario-runner" --version
    echo ""
    echo "SDK is ready to use!"
    echo ""
    echo "Example commands:"
    echo "  scenario-runner --help"
    echo "  scenario-runner --scenario <scenario.json>"
else
    echo "Warning: scenario-runner not found"
fi

# Launch shell with SDK environment
echo ""
echo "Launching SDK environment shell..."
exec $SHELL
