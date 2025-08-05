# Vulkan ML SDK Collection

This repository contains ARM ML SDK components for Vulkan, successfully ported and optimized for macOS ARM64.

## 🚀 Overview

Complete collection of ARM's ML SDK for Vulkan with comprehensive macOS ARM64 support, including:
- Full build compatibility (100% success rate)
- Apple Silicon optimizations
- Production-ready binaries
- Comprehensive documentation

## 📁 Repository Structure

This parent repository contains the following ARM ML SDK components as submodules:

- **[ai-ml-sdk-manifest](https://github.com/jerryzhao173985/ai-ml-sdk-manifest)** - Build manifest and dependency management
- **[ai-ml-sdk-for-vulkan](https://github.com/jerryzhao173985/ai-ml-sdk-for-vulkan)** - Main SDK with all fixes and optimizations
- **[ai-ml-sdk-model-converter](https://github.com/jerryzhao173985/ai-ml-sdk-model-converter)** - ML model conversion tools
- **[ai-ml-sdk-scenario-runner](https://github.com/jerryzhao173985/ai-ml-sdk-scenario-runner)** - Vulkan compute scenario execution
- **[ai-ml-sdk-vgf-library](https://github.com/jerryzhao173985/ai-ml-sdk-vgf-library)** - Vulkan Graph Format library
- **[ai-ml-emulation-layer-for-vulkan](https://github.com/jerryzhao173985/ai-ml-emulation-layer-for-vulkan)** - ARM extension emulation

## 🛠️ Quick Start

### Clone with Submodules

```bash
git clone --recursive https://github.com/jerryzhao173985/Vulkan.git
cd Vulkan
```

### Update Submodules

```bash
git submodule update --init --recursive
```

### Build the SDK

```bash
cd ai-ml-sdk-for-vulkan
./build_all_macos.sh
```

## 📋 Key Features

- ✅ **100% Build Success** - All components build successfully on macOS ARM64
- ✅ **Apple Silicon Optimized** - FP16 support, SIMD groups, unified memory
- ✅ **Production Ready** - Includes pre-built binaries and packages
- ✅ **ML Model Support** - TensorFlow Lite models included
- ✅ **Comprehensive Tools** - Performance profilers, converters, validators

## 🔧 Development Workflow

### Load Helper Functions

```bash
source git_workflow_helpers.sh
```

### Common Commands

```bash
# Check status of all repos
check_all_status

# Save all work
save_all_work "Description of changes"

# Sync with upstream ARM repos
sync_with_upstream

# Create feature branch
create_feature_branch feature/my-feature
```

## 📚 Documentation

- [Setup Instructions](SETUP_INSTRUCTIONS.md) - Initial setup guide
- [Git Workflow Guide](GIT_WORKFLOW_GUIDE.md) - Daily development workflow
- [Complete Journey Log](ai-ml-sdk-for-vulkan/COMPLETE_JOURNEY_LOG.md) - Detailed development history

## 🏆 Achievements

- Fixed 100+ compilation errors for macOS compatibility
- Resolved RAII object lifetime issues
- Added ARM extension function stubs
- Created unified ML SDK with all components
- Built production-ready package (53MB)
- Comprehensive documentation and tools

## 🔗 Links

### My Forks (with macOS fixes)
- [ai-ml-sdk-manifest](https://github.com/jerryzhao173985/ai-ml-sdk-manifest)
- [ai-ml-sdk-for-vulkan](https://github.com/jerryzhao173985/ai-ml-sdk-for-vulkan)
- [ai-ml-sdk-model-converter](https://github.com/jerryzhao173985/ai-ml-sdk-model-converter)
- [ai-ml-sdk-scenario-runner](https://github.com/jerryzhao173985/ai-ml-sdk-scenario-runner)
- [ai-ml-sdk-vgf-library](https://github.com/jerryzhao173985/ai-ml-sdk-vgf-library)
- [ai-ml-emulation-layer-for-vulkan](https://github.com/jerryzhao173985/ai-ml-emulation-layer-for-vulkan)

### Upstream ARM Repositories
- [ARM-software GitHub](https://github.com/ARM-software)

## 📄 License

This project follows the licensing terms of the original ARM ML SDK repositories. See individual submodules for specific license information.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the relevant submodule
2. Create a feature branch
3. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

---

**Built with ❤️ for macOS ARM64**