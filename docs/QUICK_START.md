# Quick Start Guide

## Installation

1. Clone the repository:
```bash
git clone --recursive https://github.com/jerryzhao173985/Vulkan.git
cd Vulkan
```

2. Run initial setup:
```bash
./vulkan-ml-sdk setup
```

## Daily Workflow

### Check Status
```bash
./vulkan-ml-sdk status
```

### Save Your Work
```bash
./vulkan-ml-sdk save "Fixed memory optimization"
```

### Sync with ARM
```bash
./vulkan-ml-sdk sync
```

### Build SDK
```bash
./vulkan-ml-sdk build
```

## All Commands

- `status` - Show repository status
- `sync` - Sync with upstream
- `save` - Commit and push changes
- `build` - Build the SDK
- `test` - Run tests
- `clean` - Clean build files
- `branch` - Create feature branch
- `info` - Show SDK information
- `help` - Show help

## Tips

1. Always check status before starting work
2. Sync regularly to avoid conflicts
3. Save work frequently
4. Use descriptive commit messages