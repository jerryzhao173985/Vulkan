# ARM ML SDK for Vulkan - Quick Start Guide

Get up and running with ML inference on Apple Silicon in under 5 minutes.

## Table of Contents

- [Quick Start](#quick-start)
- [ML Tutorials](#ml-tutorials)
- [Running Inference](#running-inference)
- [Python Tools](#python-tools)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Verify Everything Works

Run the demo script to confirm your setup:

```bash
./run_ml_demo.sh
```

This will:
- Check the SDK executable (43MB scenario-runner)
- List available ML models (7 TFLite models)
- Show compute shaders (35+ SPIR-V shaders)
- Display available Python tools

### 2. Set Up Environment

Before running any inference commands, set the library path:

```bash
export DYLD_LIBRARY_PATH=/usr/local/lib:builds/ARM-ML-SDK-Complete/lib
```

**Tip:** Add this to your `~/.zshrc` for persistence:
```bash
echo 'export DYLD_LIBRARY_PATH=/usr/local/lib:$HOME/Vulkan/builds/ARM-ML-SDK-Complete/lib' >> ~/.zshrc
```

### 3. Test the Executable

```bash
./builds/ARM-ML-SDK-Complete/bin/scenario-runner --version
```

---

## ML Tutorials

Work through these tutorials in order to understand the SDK capabilities:

### Tutorial 1: Analyze ML Models

Learn how TFLite models are structured:

```bash
./ml_tutorials/1_analyze_model.sh
```

**What you'll learn:**
- TFLite model format (TFL3)
- Model properties (input/output dimensions)
- Quantization types (INT8, FP16, FP32)
- How to inspect model files

### Tutorial 2: Test Compute Shaders

Understand Vulkan compute shaders:

```bash
./ml_tutorials/2_test_compute.sh
```

**What you'll learn:**
- Shader workgroup structure
- Buffer bindings (input_a, input_b, output)
- Dispatch dimensions
- Available shader operations

### Tutorial 3: Benchmark Operations

Measure ML operation performance:

```bash
./ml_tutorials/3_benchmark.sh
```

**What you'll learn:**
- Matrix multiplication performance
- Convolution timing
- ReLU activation throughput
- Memory bandwidth on Apple Silicon

### Tutorial 4: Style Transfer

Run neural style transfer on images:

```bash
./ml_tutorials/4_style_transfer.sh
```

**Available style models:**
| Model | Description | Size |
|-------|-------------|------|
| `la_muse` | Bright, colorful style | 7MB |
| `udnie` | Abstract, geometric patterns | 7MB |
| `mirror` | Reflective, symmetrical effects | 7MB |
| `wave_crop` | Flowing, wave-like patterns | 7MB |
| `des_glaneuses` | Classic painting style | 7MB |

### Tutorial 5: Apple Silicon Optimizations

Understand M-series performance optimizations:

```bash
./ml_tutorials/5_optimization.sh
```

**Key optimizations covered:**
- Unified Memory Architecture (zero-copy buffers)
- FP16 precision (2x throughput)
- SIMD group operations (32-thread waves)
- Pipeline caching (10x faster startup)
- Metal backend via MoltenVK

---

## Running Inference

### Basic Inference Command

```bash
./builds/ARM-ML-SDK-Complete/bin/scenario-runner \
    --scenario scenario.json \
    --output results/
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--scenario <file>` | Input scenario JSON (required) |
| `--output <dir>` | Output directory for results |
| `--profiling-dump-path <file>` | Save performance metrics |
| `--pipeline-caching` | Cache compiled shaders |
| `--cache-path <dir>` | Directory for shader cache |
| `--dry-run` | Validate without running |
| `--version` | Show version info |
| `--help` | Show all options |

### Available Models

Located in `builds/ARM-ML-SDK-Complete/models/`:

| Model | Use Case | Size |
|-------|----------|------|
| `mobilenet_v2_1.0_224_quantized_1_default_1.tflite` | Image classification | 3.4MB |
| `la_muse.tflite` | Style transfer | 7MB |
| `udnie.tflite` | Style transfer | 7MB |
| `mirror.tflite` | Style transfer | 7MB |
| `wave_crop.tflite` | Style transfer | 7MB |
| `des_glaneuses.tflite` | Style transfer | 7MB |
| `fire_detection.tflite` | Fire detection | 8.1MB |

### Creating a Scenario File

**Example: Image Classification**

```json
{
  "name": "Image Classification",
  "model_path": "builds/ARM-ML-SDK-Complete/models/mobilenet_v2_1.0_224_quantized_1_default_1.tflite",
  "input": {
    "type": "image",
    "width": 224,
    "height": 224,
    "format": "RGB"
  },
  "preprocessing": [
    {"operation": "resize", "width": 224, "height": 224},
    {"operation": "normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
  ],
  "inference": {
    "backend": "vulkan",
    "precision": "int8"
  },
  "output": {
    "type": "classification",
    "top_k": 5
  }
}
```

**Example: Style Transfer**

```json
{
  "name": "Style Transfer",
  "model_path": "builds/ARM-ML-SDK-Complete/models/la_muse.tflite",
  "input": {
    "type": "image",
    "width": 256,
    "height": 256,
    "format": "RGB"
  },
  "preprocessing": [
    {"operation": "resize", "width": 256, "height": 256},
    {"operation": "normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
  ],
  "inference": {
    "backend": "vulkan",
    "precision": "fp16"
  },
  "postprocessing": [
    {"operation": "denormalize"},
    {"operation": "clip", "min": 0, "max": 255}
  ],
  "output": {
    "type": "image",
    "format": "RGB",
    "save_path": "styled_output.jpg"
  }
}
```

**Example: Compute Shader Test**

```json
{
  "name": "Vector Addition",
  "compute_operations": [
    {
      "shader": "add",
      "workgroup_size": [64, 1, 1],
      "dispatch": [1, 1, 1],
      "buffers": [
        {"name": "input_a", "size": 256, "data": "random"},
        {"name": "input_b", "size": 256, "data": "random"},
        {"name": "output", "size": 256, "usage": "storage"}
      ]
    }
  ]
}
```

---

## Python Tools

Located in `builds/ARM-ML-SDK-Complete/tools/`:

### analyze_tflite_model.py

Inspect TFLite model structure and generate Vulkan pipelines:

```bash
python3 builds/ARM-ML-SDK-Complete/tools/analyze_tflite_model.py \
    builds/ARM-ML-SDK-Complete/models/la_muse.tflite \
    --output-dir ./pipelines \
    --verbose
```

**Features:**
- Extracts model operations (CONV_2D, RELU, etc.)
- Identifies tensor shapes and types
- Generates Vulkan pipeline JSON
- Maps TFLite ops to compute shaders

### optimize_for_apple_silicon.py

Apply M-series specific optimizations:

```bash
python3 builds/ARM-ML-SDK-Complete/tools/optimize_for_apple_silicon.py
```

**Optimizations applied:**
- FP16 acceleration
- SIMD group operations (32-thread waves)
- Optimized tile sizes (32x32)
- Winograd algorithm for 3x3 convolutions

### profile_performance.py

Benchmark and visualize ML operations:

```bash
python3 builds/ARM-ML-SDK-Complete/tools/profile_performance.py
```

**Output:**
- Per-operation timing (ms)
- Success/failure status
- Performance visualization (PNG chart)

### Additional Tools

| Tool | Description |
|------|-------------|
| `convert_model_optimized.py` | Convert models with optimizations |
| `create_ml_pipeline.py` | Create Vulkan compute pipelines |
| `validate_ml_operations.py` | Validate operation compatibility |
| `realtime_performance_monitor.py` | Live performance monitoring |

---

## Advanced Usage

### Performance Profiling

Capture detailed performance metrics:

```bash
./builds/ARM-ML-SDK-Complete/bin/scenario-runner \
    --scenario test.json \
    --output results/ \
    --profiling-dump-path profile.json
```

The profile JSON contains:
- Per-operation timing
- Memory allocation stats
- GPU utilization
- Pipeline stage durations

### Pipeline Caching

Speed up repeated runs with shader caching:

```bash
./builds/ARM-ML-SDK-Complete/bin/scenario-runner \
    --scenario test.json \
    --pipeline-caching \
    --cache-path /tmp/shader_cache/
```

**Benefits:**
- ~10x faster pipeline creation after first run
- Compiled shaders cached to disk
- Instant startup on subsequent runs

### Available Compute Shaders

35 SPIR-V shaders in `builds/ARM-ML-SDK-Complete/shaders/`:

**Basic Operations:**
- `add.spv`, `multiply.spv`, `sub_shader.spv`

**ML Operations:**
- `conv1d_fixed.spv`, `optimized_conv2d.spv`
- `matrix_multiply.spv`
- `relu.spv`, `sigmoid.spv`

**Tensor Operations:**
- `tensor.spv`, `tensor_shader.spv`
- `copy_tensor_shader.spv`
- `plus_ten_tensor.spv`

**Image Processing:**
- `image_shader.spv`, `copy_img_shader.spv`
- `passthrough_*.spv` (various formats)

### Custom Shader Integration

Use your own SPIR-V shaders:

```json
{
  "compute_operations": [
    {
      "shader": "path/to/your/shader.spv",
      "workgroup_size": [256, 1, 1],
      "dispatch": [1024, 1, 1],
      "buffers": [...]
    }
  ]
}
```

### Batch Processing

Process multiple inputs efficiently:

```bash
for img in images/*.jpg; do
    ./builds/ARM-ML-SDK-Complete/bin/scenario-runner \
        --scenario style_transfer.json \
        --input "$img" \
        --output "styled_$(basename $img)"
done
```

---

## Performance on Apple Silicon

Expected performance on M4 Max:

| Operation | Time | Details |
|-----------|------|---------|
| Conv2D | 2.5ms | 224x224x32 |
| MatMul | 1.2ms | 1024x1024 |
| Style Transfer | 150ms | 256x256 image |
| Memory Bandwidth | 400GB/s | Unified memory |

**Optimization Tips:**
1. Use FP16 for inference (2x faster than FP32)
2. Batch operations to reduce dispatch overhead
3. Align buffers to 256 bytes
4. Enable pipeline caching
5. Profile with `--profiling-dump-path`

---

## Troubleshooting

### Library Not Found

```
dyld: Library not loaded: libVGF.dylib
```

**Fix:** Set the library path:
```bash
export DYLD_LIBRARY_PATH=/usr/local/lib:builds/ARM-ML-SDK-Complete/lib
```

### Vulkan Device Not Found

```
Error: Failed to find a Vulkan device
```

**Fix:** Ensure MoltenVK is installed:
```bash
brew install molten-vk
```

### Model Format Error

```
Error: Unknown TFLite version
```

**Fix:** Ensure model is TFLite v3 format. Re-export from TensorFlow if needed.

### Out of Memory

```
Error: VK_ERROR_OUT_OF_DEVICE_MEMORY
```

**Fix:** Reduce batch size or model precision:
```json
{
  "inference": {
    "precision": "fp16",
    "batch_size": 1
  }
}
```

---

## Next Steps

1. **Explore the demos:** `./run_ml_demo.sh`
2. **Work through tutorials:** `./ml_tutorials/1_analyze_model.sh`
3. **Try your own models:** Convert with `analyze_tflite_model.py`
4. **Optimize performance:** Use FP16 and pipeline caching
5. **Profile your workloads:** Enable `--profiling-dump-path`

For complete technical details, see [CLAUDE.md](CLAUDE.md).

---

**SDK Version:** ARM ML SDK for Vulkan (Production)
**Platform:** macOS ARM64 (M-series)
**Status:** Ready for ML workloads
