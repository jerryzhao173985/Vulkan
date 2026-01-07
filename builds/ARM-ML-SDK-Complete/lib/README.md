# ARM ML SDK Libraries

This directory contains the libraries required by the ARM ML SDK for Vulkan.

## Static Libraries (7 SPIRV)

- `libSPIRV.a` - Core SPIR-V library
- `libSPIRV-Tools.a` - SPIR-V utilities
- `libSPIRV-Tools-opt.a` - SPIR-V optimizer (largest, ~655MB)
- `libSPIRV-Tools-link.a` - SPIR-V linker
- `libSPIRV-Tools-reduce.a` - SPIR-V reducer
- `libSPIRV-Tools-diff.a` - SPIR-V diff tool
- `libSPIRV-Tools-lint.a` - SPIR-V linter

## Shared Libraries

- `libSPIRV-Tools-shared.dylib` - Shared SPIR-V tools library

## VGF Library Note

The VGF (Vulkan Graph Format) library is **statically linked** into the `scenario-runner`
executable during the SDK build process. It is not distributed as a separate `.a` file.

VGF source code is available in the `ai-ml-sdk-vgf-library/` submodule and provides:
- VGF encoder/decoder for ML graph serialization
- C and C++ APIs for graph manipulation
- Support for TFLite model conversion

## Installation

Library files (*.a, *.dylib) are excluded from version control.
These libraries are copied from `/usr/local/lib/` during SDK setup.

To populate this directory, run:
```bash
cp /usr/local/lib/libSPIRV*.a .
cp /usr/local/lib/libSPIRV-Tools-shared.dylib .
```

## Prerequisites

Install SPIRV-Tools via Homebrew:
```bash
brew install spirv-tools
```
