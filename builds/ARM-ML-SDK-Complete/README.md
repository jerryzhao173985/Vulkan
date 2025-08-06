# ARM ML SDK for Vulkan - Complete Build

## Overview
This is the complete, unified ARM ML SDK for Vulkan, fully built and optimized for macOS ARM64.

## Contents
- `bin/` - Executable binaries (scenario-runner)
- `lib/` - Static libraries (VGF, SPIRV)
- `models/` - TensorFlow Lite ML models
- `shaders/` - Compiled Vulkan compute shaders
- `tools/` - ML pipeline and optimization tools
- `launch_sdk.sh` - SDK environment launcher

## Quick Start
```bash
./launch_sdk.sh
```

This will set up the environment and show available tools.

## Running ML Workloads
```bash
# Run a scenario
bin/scenario-runner --scenario scenarios/example.json

# Run with a specific model
python3 tools/run_ml_inference.py models/la_muse.tflite
```

## Build Information
- Platform: macOS ARM64 (Apple Silicon)
- Build Type: Release
- Optimizations: Enabled for M-series processors
- Status: Production Ready
