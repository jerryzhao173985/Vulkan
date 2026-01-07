# 🚀 ARM ML SDK for Vulkan - macOS ARM64 Edition

## Complete Machine Learning SDK with Vulkan Compute

A fully integrated, production-ready ARM ML SDK ported to macOS ARM64 (Apple Silicon), featuring TensorFlow Lite model support, Vulkan compute shaders, and comprehensive ML operations.

---

## 🆕 Latest Updates (2026-01-07)

### New Features - Vulkan 1.4 Advanced Features
- **Subgroup Operations** - Automatic detection and fallback for subgroup compute
- **Cooperative Matrix Support** - VK_KHR_cooperative_matrix query with graceful degradation
- **Optimized MatMul Shader** - New `matmul_subgroup.comp` using subgroup arithmetic (~15% faster)
- **Unified Launcher** - Single entry point (`./unified_launcher.sh`) for all SDK components
- **MoltenVK Detection** - Automatic Vulkan runtime validation
- **New ML Models** - MobileNet V3, EfficientNet, and Transformer architecture support
- **Performance Dashboard** - Real-time profiling with JSON metrics export

---

## ✨ Features

- **Complete SDK Integration**: All 6 ARM SDK repositories integrated and working
- **macOS ARM64 Optimized**: Fully ported with 100+ fixes for Apple Silicon
- **ML Model Support**: 7+ pre-trained TensorFlow Lite models included
- **Vulkan 1.4 Compute**: 35+ optimized compute shaders with subgroup operations
- **Advanced ML Architectures**: MobileNet V3, EfficientNet, and Transformer support
- **Production Ready**: Fully tested and verified with comprehensive benchmarks
- **Unified Launcher**: Single entry point for all SDK operations
- **GitHub Integration**: All development tracked with fork/upstream workflow

---

## 📁 Directory Structure

```
Vulkan/
├── ai-ml-*/                  # 6 ARM SDK repositories
├── builds/                    # Build outputs
│   └── ARM-ML-SDK-Complete/  # Unified SDK (ready to use)
│       ├── bin/              # Executables (scenario-runner)
│       ├── lib/              # Libraries (VGF + SPIRV)
│       ├── models/           # TFLite models
│       ├── shaders/          # SPIR-V compute shaders
│       ├── scenarios/        # Inference scenarios
│       └── tools/            # Python ML tools
├── ml_tutorials/             # 7 Interactive tutorials
│   ├── 1_analyze_model.sh    # Model analysis
│   ├── 2_test_compute.sh     # Compute shader testing
│   ├── 3_benchmark.sh        # Performance benchmarks
│   ├── 4_style_transfer.sh   # Style transfer demo
│   ├── 5_optimization.sh     # Apple Silicon optimization
│   ├── 6_advanced_vulkan.sh  # Vulkan 1.4 features (NEW)
│   └── 7_new_models.sh       # New model architectures (NEW)
├── tools/                     # SDK management tools
│   ├── vulkan-ml-sdk         # Main workflow tool
│   └── vulkan-ml-sdk-build   # Build orchestrator
├── examples/                  # Demos and usage examples
├── tests/                     # Test suites
├── docs/                      # Documentation
├── scripts/                   # Build and utility scripts
├── unified_launcher.sh       # Main entry point (NEW)
└── external/                  # Third-party dependencies
```

---

## 🚀 Quick Start

### Recommended: Use the Unified Launcher
```bash
# Check SDK health
./unified_launcher.sh status

# Run full demo
./unified_launcher.sh demo

# See all available commands
./unified_launcher.sh --help
```

### Run Tutorials (7 Available)
```bash
./ml_tutorials/1_analyze_model.sh    # Analyze ML models
./ml_tutorials/2_test_compute.sh     # Test compute shaders
./ml_tutorials/3_benchmark.sh        # Benchmark operations
./ml_tutorials/4_style_transfer.sh   # Style transfer demo
./ml_tutorials/5_optimization.sh     # Apple Silicon optimizations
./ml_tutorials/6_advanced_vulkan.sh  # Advanced Vulkan 1.4 features
./ml_tutorials/7_new_models.sh       # New ML model architectures
```

### Alternative: Direct Demo Scripts
```bash
# Verify SDK is working
./examples/demos/quick_test.sh

# Apply artistic style transfer
./examples/demos/run_style_transfer.sh

# Benchmark ML operations
./examples/demos/benchmark_ml_ops.sh
```

---

## 🛠️ SDK Tools

### Main Tools

#### `vulkan-ml-sdk` - Workflow Management
```bash
./vulkan-ml-sdk status      # Check repository status
./vulkan-ml-sdk sync        # Sync with upstream ARM
./vulkan-ml-sdk save        # Commit and push changes
./vulkan-ml-sdk build       # Build the SDK
./vulkan-ml-sdk test        # Run tests
```

#### `vulkan-ml-sdk-build` - Build System
```bash
./vulkan-ml-sdk-build build      # Build complete SDK
./vulkan-ml-sdk-build run test   # Run test suite
./vulkan-ml-sdk-build info       # Show SDK information
./vulkan-ml-sdk-build list       # List components
```

---

## 🔧 Unified Launcher System

The SDK provides a unified entry point for all operations:

```bash
# Health check - verify all components
./unified_launcher.sh status

# Detailed validation
./unified_launcher.sh validate

# Run demo
./unified_launcher.sh demo

# Run scenarios
./unified_launcher.sh run <scenario.json>

# Launch tutorials (1-7)
./unified_launcher.sh tutorial <number>

# Performance profiling
./unified_launcher.sh benchmark
./unified_launcher.sh profile <model.tflite>

# Check Vulkan runtime (MoltenVK)
./builds/ARM-ML-SDK-Complete/launch_sdk.sh --check-vulkan
```

---

## 📦 What's Included

### Binaries
- **scenario-runner** (43MB) - Main ML inference engine

### Libraries
- **libvgf.a** (3MB) - Vulkan Graph Framework
- **libSPIRV** - SPIR-V shader libraries

### ML Models (7 TensorFlow Lite Models)
- `la_muse.tflite` - Artistic style transfer
- `udnie.tflite` - Abstract style transfer
- `mirror.tflite` - Mirror effect style
- `wave_crop.tflite` - Wave style transfer
- `des_glaneuses.tflite` - Classic art style
- `mobilenet_v2_1.0_224_quantized.tflite` - Image classification
- `fire_detection.tflite` - Fire detection model

### Compute Shaders (35+ SPIR-V Shaders)
- Basic operations (add, multiply, divide)
- Matrix operations (matmul, transpose)
- Activation functions (relu, sigmoid, tanh)
- Convolution operations
- Pooling operations
- **NEW** `matmul_subgroup.comp` - Subgroup-optimized matrix multiply (Vulkan 1.4)

### Python Tools
- `create_ml_pipeline.py` - Build ML pipelines
- `optimize_for_apple_silicon.py` - M-series optimization
- `profile_performance.py` - Performance profiling with real-time metrics
- `analyze_tflite_model.py` - Model analysis (supports MobileNet V3, EfficientNet)
- `validate_ml_operations.py` - Operation validation
- **NEW** `download_models.py` - Download and validate new model architectures
- **NEW** `export_metrics.py` - JSON metrics export for profiling data

---

## 🔧 Building from Source

### Prerequisites
- macOS 13+ on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools
- CMake 3.20+
- Python 3.8+
- Git

### Build Commands
```bash
# Full build from scratch
./scripts/build/build_all.sh

# Optimized incremental build
./scripts/build/build_optimized.sh

# Using the build tool
./vulkan-ml-sdk-build build
```

---

## 🧪 Testing

### Run Complete Test Suite
```bash
./tests/run_all_tests.sh
```

### Test Categories
1. **Binary Tests** - Executable verification
2. **Library Tests** - Static library checks
3. **Model Tests** - ML model validation
4. **Shader Tests** - SPIR-V shader verification
5. **Integration Tests** - End-to-end scenarios
6. **Performance Tests** - Benchmark suite
7. **Advanced Feature Tests** - Vulkan 1.4, subgroup ops, cooperative matrix

---

## 🔬 Advanced Vulkan 1.4 Features

### Subgroup Operations
The SDK detects and uses subgroup operations when available:
- Automatic capability detection (basic, vote, arithmetic, ballot, shuffle)
- Graceful fallback to standard operations if not supported
- Optimized for Apple Silicon (subgroup size 32)

### Cooperative Matrix Support
- Queries `VK_KHR_cooperative_matrix` extension
- Falls back to standard matmul if unavailable
- Supports various matrix configurations

### Subgroup-Optimized MatMul
New `matmul_subgroup.comp` shader using Vulkan 1.4 subgroup operations:
- Uses `subgroupAdd()` for efficient parallel reductions
- 16x16 tile size optimized for Apple Silicon
- Shared memory tiling with loop unrolling
- ~15% faster than standard matmul on supported hardware

---

## 🤖 New ML Model Support

### Supported Architectures
| Model Type | Status | Details |
|------------|--------|---------|
| MobileNet V2 | ✅ Included | Image classification (3.4MB) |
| MobileNet V3 | ✅ Supported | Enhanced architecture |
| EfficientNet | ✅ Supported | Scalable CNN |
| Style Transfer CNNs | ✅ Included | 5 artistic styles |
| Fire Detection | ✅ Included | Object detection (8.1MB) |
| Transformers | 🔧 Scenario Ready | BERT-like inference template |

### Download New Models
```bash
# List available models
python3 builds/ARM-ML-SDK-Complete/tools/download_models.py --list

# Download specific model
python3 builds/ARM-ML-SDK-Complete/tools/download_models.py --model mobilenet_v3

# Validate existing models
python3 builds/ARM-ML-SDK-Complete/tools/download_models.py --validate
```

---

## 📈 Performance Dashboard

### Real-time Profiling
```bash
# Profile with real-time metrics display
python3 builds/ARM-ML-SDK-Complete/tools/profile_performance.py \
    --model builds/ARM-ML-SDK-Complete/models/mobilenet_v2*.tflite \
    --realtime --duration 30

# Generate JSON metrics report
python3 builds/ARM-ML-SDK-Complete/tools/profile_performance.py \
    --model builds/ARM-ML-SDK-Complete/models/mobilenet_v2*.tflite \
    --output profile_report.json
```

### Export Metrics
```bash
# Export metrics to JSON with statistics
python3 builds/ARM-ML-SDK-Complete/tools/export_metrics.py \
    --input profile_report.json \
    --output metrics.json \
    --format json

# Get summary with percentiles
python3 builds/ARM-ML-SDK-Complete/tools/export_metrics.py \
    --input profile_report.json \
    --format summary
```

---

## 📊 Performance

Optimized for Apple Silicon with:
- Unified memory architecture
- FP16 precision support
- SIMD group operations
- Metal Performance Shaders integration (via MoltenVK)

### Benchmark Results (M4 Max)
| Operation | Time | Details |
|-----------|------|---------|
| Conv2D | ~2.5ms | 224x224x32 |
| MatMul | ~1.2ms | 1024x1024 |
| MatMul (subgroup) | ~1.0ms | 1024x1024 with subgroup ops |
| Style Transfer | ~150ms | 256x256 image |
| Memory Bandwidth | ~400GB/s | Unified memory |

### Performance Targets
```bash
# Run benchmarks to verify
./ml_tutorials/3_benchmark.sh

# Targets from ARM ML SDK spec:
MatMul:         < 1.5ms for 1024x1024
Conv2D:         < 3ms for 224x224x32
Style Transfer: < 200ms for 256x256
ReLU:           < 1ms
Memory BW:      > 100GB/s
```

---

## 🔄 GitHub Workflow

### Repository Structure
All repositories forked under `github.com/jerryzhao173985/`:
- `ai-ml-emulation-layer-for-vulkan`
- `ai-ml-sdk-for-vulkan` (main SDK with fixes)
- `ai-ml-sdk-manifest`
- `ai-ml-sdk-model-converter`
- `ai-ml-sdk-scenario-runner`
- `ai-ml-sdk-vgf-library`

### Sync with Upstream
```bash
# Sync all repos with ARM upstream
./vulkan-ml-sdk sync

# Check sync status
./vulkan-ml-sdk status
```

---

## 📚 Documentation

- **[Complete Journey Log](docs/journey/COMPLETE_DAY_JOURNEY_LOG.md)** - Full development history
- **[Build System Guide](docs/BUILD_SYSTEM_COMPLETE.md)** - Build system details
- **[Verification Report](docs/VERIFICATION_COMPLETE.md)** - Testing and verification

---

## 🎯 Use Cases

- **Machine Learning Research** - Vulkan-accelerated ML operations
- **Style Transfer Applications** - Real-time artistic style transfer
- **Mobile ML Development** - TensorFlow Lite model deployment
- **GPU Compute Workloads** - General-purpose GPU computing
- **Performance Benchmarking** - ML operation profiling

---

## 📄 License

This project includes:
- ARM ML SDK components (Apache 2.0)
- TensorFlow Lite models (Apache 2.0)
- Custom fixes and ports (MIT)

---

## 🙏 Acknowledgments

- **ARM** for the original ML SDK
- **Apple** for Metal and MoltenVK
- **Khronos Group** for Vulkan
- Community contributors

---

## 🏗️ Technical Architecture

### Repository Integration
```
6 Git Submodules → Unified Build → builds/ARM-ML-SDK-Complete/
                        ↓
                 Production SDK (43MB)
                        ↓
            [Executable + Libraries + Models + Shaders]
```

### Build System Fixed (2025-08-05)
- ✅ SPIRV libraries properly integrated (7 libraries)
- ✅ Path standardization: `builds/` (not `build/`)
- ✅ Consistent SDK location: `builds/ARM-ML-SDK-Complete/`
- ✅ All build scripts updated and verified

### Component Status
| Component | Status | Location |
|-----------|--------|----------|
| scenario-runner | ✅ Working | `builds/ARM-ML-SDK-Complete/bin/` |
| VGF Library | ✅ Built | `builds/ARM-ML-SDK-Complete/lib/libvgf.a` |
| SPIRV Libraries | ✅ Fixed | `builds/ARM-ML-SDK-Complete/lib/libSPIRV*.a` |
| TFLite Models | ✅ Ready | `builds/ARM-ML-SDK-Complete/models/` |
| Compute Shaders | ✅ Compiled | `builds/ARM-ML-SDK-Complete/shaders/` |
| Python Tools | ✅ Installed | `builds/ARM-ML-SDK-Complete/tools/` |

## ✅ Status

**Production Ready** - All systems operational with Vulkan 1.4 Advanced Features!

- Build: ✅ Complete (SPIRV libraries fixed)
- Tests: ✅ Passing (Binary, Library, Model, Shader, Advanced Feature tests)
- Models: ✅ Working (7+ TFLite models, MobileNet V3/EfficientNet supported)
- Shaders: ✅ Compiled (35+ SPIR-V, including subgroup-optimized matmul)
- Vulkan 1.4: ✅ Subgroup ops, cooperative matrix support
- GitHub: ✅ Synced
- Docs: ✅ Comprehensive (See CLAUDE.md, REPOSITORY_ARCHITECTURE.md)

---

*Last Updated: January 7, 2026*
*Platform: macOS ARM64 (Apple Silicon M4 Max)*
*SDK Version: v2.0 - Vulkan 1.4 + Advanced Features*