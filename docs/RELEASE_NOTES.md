# 🚀 ARM ML SDK for Vulkan - Release Notes

## Version 2.0.0 (2026-01-07)

**Status:** Production Ready
**Platform:** macOS ARM64 (Apple Silicon M-series)
**Vulkan Version:** 1.4 with Advanced Features

---

## 📋 Release Summary

This major release introduces Vulkan 1.4 advanced features, a unified launcher system, new ML model architectures, and comprehensive performance profiling capabilities. The SDK is now production-ready for ML workloads on Apple Silicon.

---

## ✨ New Features

### Vulkan 1.4 Advanced Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Subgroup Operations** | Automatic detection and fallback for subgroup compute | ✅ Production |
| **Cooperative Matrix** | VK_KHR_cooperative_matrix query with graceful degradation | ✅ Production |
| **Subgroup MatMul** | New `matmul_subgroup.comp` using subgroup arithmetic | ✅ Production |

#### Subgroup Operations
The SDK now automatically detects and utilizes subgroup operations when available:
- `GL_KHR_shader_subgroup_basic`
- `GL_KHR_shader_subgroup_vote`
- `GL_KHR_shader_subgroup_arithmetic`
- `GL_KHR_shader_subgroup_ballot`
- `GL_KHR_shader_subgroup_shuffle`

Falls back gracefully to standard operations on unsupported hardware.

#### Cooperative Matrix Support
Queries `VK_KHR_cooperative_matrix` extension availability with automatic fallback to standard matrix operations when unavailable.

### Unified Launcher System

New single entry point (`./unified_launcher.sh`) for all SDK operations:

```bash
./unified_launcher.sh status     # Check SDK health
./unified_launcher.sh demo       # Run full demo
./unified_launcher.sh validate   # Detailed validation
./unified_launcher.sh run <file> # Run scenarios
./unified_launcher.sh tutorial <1-7>  # Launch tutorials
./unified_launcher.sh benchmark  # Performance profiling
```

### New ML Model Support

| Model Type | Status | Details |
|------------|--------|---------|
| MobileNet V2 | ✅ Included | Image classification (3.4MB) |
| MobileNet V3 | ✅ Supported | Enhanced architecture |
| EfficientNet | ✅ Supported | Scalable CNN |
| Style Transfer CNNs | ✅ Included | 5 artistic styles |
| Fire Detection | ✅ Included | Object detection (8.1MB) |
| Transformers | 🔧 Scenario Ready | BERT-like inference template |

### Performance Dashboard

Real-time profiling with JSON metrics export:
- `profile_performance.py` - Performance analysis with real-time metrics
- `export_metrics.py` - JSON metrics export for profiling data
- `download_models.py` - Download and validate new model architectures

### MoltenVK Detection

Automatic Vulkan runtime validation with clear error messages and installation instructions when MoltenVK is not found.

---

## 🔧 Improvements

### Build System

- ✅ SPIRV libraries now properly copied to `builds/ARM-ML-SDK-Complete/lib/`
- ✅ Standardized all paths to use `builds/` directory
- ✅ Fixed `build_optimized.sh` and `build_all.sh` for consistent SDK creation
- ✅ All 7 SPIRV libraries present and working
- ✅ ARM-staging branch support for SPIRV-Tools (required for ARM ML extensions)
- ✅ Apple Silicon M-series optimization flags (`-mcpu=apple-m1`, `-O3`, `-fvectorize`)

### Compute Shaders

- New `matmul_subgroup.comp` - Subgroup-optimized matrix multiply
  - Uses `subgroupAdd()` for efficient parallel reductions
  - 16x16 tile size optimized for Apple Silicon
  - Shared memory tiling with loop unrolling
  - ~15% faster than standard matmul on supported hardware
- Total shaders: 35+ SPIR-V compiled shaders

### Python Tools

New tools added:
- `download_models.py` - Download and validate new model architectures
- `export_metrics.py` - JSON metrics export for profiling data

Updated tools:
- `analyze_tflite_model.py` - Now supports MobileNet V3 and EfficientNet
- `profile_performance.py` - Added real-time metrics display

### Tutorials

New tutorials added:
- `ml_tutorials/6_advanced_vulkan.sh` - Advanced Vulkan 1.4 features
- `ml_tutorials/7_new_models.sh` - New ML model architectures

---

## 📊 Performance Benchmarks (M4 Max)

| Operation | Time | Details |
|-----------|------|---------|
| Conv2D | 2.5ms | 224x224x32 |
| MatMul | 1.2ms | 1024x1024 |
| MatMul (subgroup) | ~1.0ms | 1024x1024 with subgroup ops |
| Style Transfer | 150ms | 256x256 image |
| Memory Bandwidth | 400GB/s | Unified memory |

### Performance Targets

```
MatMul:         < 1.5ms for 1024x1024
Conv2D:         < 3ms for 224x224x32
Style Transfer: < 200ms for 256x256
ReLU:           < 1ms
Memory BW:      > 100GB/s
```

---

## 🧪 Test Results

```
Binary Tests:           ✅ PASS (4/4)
Library Tests:          ✅ PASS (2/2) - VGF and SPIRV libraries
Model Tests:            ✅ PASS (7/7) - All TFLite models
Shader Tests:           ✅ PASS (3/3) - 35+ shaders compiled
Advanced Feature Tests: ✅ PASS - Vulkan 1.4, subgroup, cooperative matrix
```

### Test Suite

Run all tests with:
```bash
./tests/run_all_tests.sh
```

End-to-end pipeline test:
```bash
./tests/test_e2e_pipeline.sh
```

---

## 📦 Package Contents

### SDK Components

| Component | Count/Size | Location |
|-----------|------------|----------|
| scenario-runner | 43MB | `builds/ARM-ML-SDK-Complete/bin/` |
| VGF Library | 3.0MB | `builds/ARM-ML-SDK-Complete/lib/libvgf.a` |
| SPIRV Libraries | 7 libs | `builds/ARM-ML-SDK-Complete/lib/` |
| ML Models | 7 models (46MB) | `builds/ARM-ML-SDK-Complete/models/` |
| Compute Shaders | 35+ shaders | `builds/ARM-ML-SDK-Complete/shaders/` |
| Python Tools | 9 tools | `builds/ARM-ML-SDK-Complete/tools/` |
| Scenarios | Multiple | `builds/ARM-ML-SDK-Complete/scenarios/` |

### Included ML Models

- `mobilenet_v2_1.0_224_quantized_1_default_1.tflite` (3.4MB)
- `la_muse.tflite` (7.0MB)
- `udnie.tflite` (7.0MB)
- `mirror.tflite` (7.0MB)
- `wave_crop.tflite` (7.0MB)
- `des_glaneuses.tflite` (7.0MB)
- `fire_detection.tflite` (8.1MB)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jerryzhao173985/ai-ml-sdk-for-vulkan.git
cd ai-ml-sdk-for-vulkan

# Build the SDK
python3 scripts/build.py --build-type Release --threads 8
```

### Environment Setup

```bash
export DYLD_LIBRARY_PATH=/usr/local/lib:builds/ARM-ML-SDK-Complete/lib
export PATH=$PWD/builds/ARM-ML-SDK-Complete/bin:$PATH
export VK_LAYER_PATH=$PWD/builds/ARM-ML-SDK-Complete/lib
```

### Verification

```bash
# Check SDK status
./unified_launcher.sh status

# Run demo
./unified_launcher.sh demo

# Run scenario
./builds/ARM-ML-SDK-Complete/bin/scenario-runner --version
```

---

## ⚠️ Breaking Changes

None in this release. All existing scripts and APIs remain compatible.

---

## 🐛 Bug Fixes

### Build System (from v1.x)
- Fixed SPIRV library copy issues in build scripts
- Resolved path inconsistencies (`build/` vs `builds/`)
- Fixed Apple Silicon optimization flag compatibility

### Runtime
- RAII object lifetime fixes (placement new pattern) - 42+ instances
- Namespace qualification fixes for vk:: classes
- Container operation fixes (emplace with piecewise_construct)

---

## ⚙️ Known Issues

1. **Cooperative Matrix**: Not all hardware supports `VK_KHR_cooperative_matrix`. The SDK falls back to standard operations automatically.

2. **Transformer Models**: Currently at "Scenario Ready" status. Full inference support requires scenario template customization.

3. **NumPy Compatibility**: Requires `numpy<2.0` due to breaking API changes in NumPy 2.0.

---

## 📋 Requirements

### System Requirements

- **OS:** macOS 12.0+ (Monterey or later)
- **Hardware:** Apple Silicon (M1/M2/M3/M4 series)
- **Vulkan:** MoltenVK runtime installed
- **Memory:** 8GB+ RAM recommended

### Dependencies

- Python 3.9+
- `numpy<2.0`
- `Pillow==11.0.0`
- `pybind11`
- CMake 3.20+
- Xcode Command Line Tools

---

## 📚 Documentation

- `docs/QUICK_START.md` - Getting started guide
- `docs/BUILD_SYSTEM_COMPLETE.md` - Build system documentation
- `CLAUDE.md` - Technical progress report
- `REPOSITORY_ARCHITECTURE.md` - Complete architecture details

---

## 🙏 Acknowledgments

This release builds upon the work of:
- ARM ML team for the original SDK
- Khronos Group for Vulkan 1.4 specifications
- MoltenVK team for macOS Vulkan support
- SPIRV-Tools contributors (ARM-staging branch)

---

## 📝 Changelog

### v2.0.0 (2026-01-07)
- Added Vulkan 1.4 subgroup operations support
- Added cooperative matrix detection with fallback
- New unified launcher system
- New ML model architectures (MobileNet V3, EfficientNet, Transformers)
- Performance dashboard with JSON metrics export
- MoltenVK detection and validation
- 7 tutorials including advanced features
- 22 new test cases
- End-to-end pipeline testing

### v1.0.0 (2025-08-05)
- Initial production release
- Complete ARM ML SDK port to macOS ARM64
- 100+ compilation fixes
- 6 ARM SDK repositories integrated
- 7 TFLite models included
- 35 SPIR-V shaders compiled
- Build system with Apple Silicon optimizations

---

**Full documentation:** See `CLAUDE.md` for complete technical details.

**Report issues:** https://github.com/jerryzhao173985/ai-ml-sdk-for-vulkan/issues
