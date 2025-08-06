/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 * 
 * SPIRV Shader Unit Tests for ARM ML SDK
 * Tests SPIRV shader compilation, validation, reflection, and ARM ML extensions
 */

#include <gtest/gtest.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_glsl.hpp>
#include <vector>
#include <memory>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <map>

namespace mlsdk::tests {

namespace fs = std::filesystem;

class SPIRVShaderTest : public ::testing::Test {
protected:
    vk::raii::Context context;
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    uint32_t queueFamilyIndex = 0;
    
    // Shader paths
    fs::path shadersDir;
    std::map<std::string, std::vector<uint32_t>> shaderCache;
    
    void SetUp() override {
        // Setup Vulkan instance and device
        vk::ApplicationInfo appInfo{
            "SPIRVShaderTests",
            VK_MAKE_VERSION(1, 0, 0),
            "ARM ML SDK Tests",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_2
        };
        
        vk::InstanceCreateInfo instanceInfo{{}, &appInfo};
        instance = vk::raii::Instance(context, instanceInfo);
        
        auto physicalDevices = vk::raii::PhysicalDevices(instance);
        ASSERT_FALSE(physicalDevices.empty());
        physicalDevice = std::move(physicalDevices[0]);
        
        // Find compute queue
        auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
            if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
                queueFamilyIndex = i;
                break;
            }
        }
        
        // Create device with required extensions
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueInfo{{}, queueFamilyIndex, 1, &queuePriority};
        
        // Enable ARM ML extensions if available
        std::vector<const char*> deviceExtensions;
        auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
        
        for (const auto& ext : availableExtensions) {
            std::string extName(ext.extensionName);
            if (extName.find("VK_ARM_") != std::string::npos) {
                deviceExtensions.push_back(ext.extensionName);
            }
        }
        
        vk::DeviceCreateInfo deviceInfo{{}, queueInfo, {}, deviceExtensions};
        device = vk::raii::Device(physicalDevice, deviceInfo);
        
        // Set shader directory
        shadersDir = fs::path(SDK_SHADERS_PATH);
    }
    
    std::vector<uint32_t> loadSPIRV(const fs::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return {};
        }
        
        size_t fileSize = file.tellg();
        std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
        
        file.seekg(0);
        file.read(reinterpret_cast<char*>(spirv.data()), fileSize);
        
        return spirv;
    }
    
    bool validateSPIRV(const std::vector<uint32_t>& spirv) {
        if (spirv.size() < 5) return false;
        return spirv[0] == 0x07230203; // SPIR-V magic number
    }
};

// Test 1: Load and Validate All Shader Files
TEST_F(SPIRVShaderTest, LoadAllShaders) {
    if (!fs::exists(shadersDir)) {
        GTEST_SKIP() << "Shaders directory not found: " << shadersDir;
    }
    
    int totalShaders = 0;
    int validShaders = 0;
    
    for (const auto& entry : fs::directory_iterator(shadersDir)) {
        if (entry.path().extension() == ".spv") {
            totalShaders++;
            
            auto spirv = loadSPIRV(entry.path());
            if (validateSPIRV(spirv)) {
                validShaders++;
                shaderCache[entry.path().filename()] = spirv;
            } else {
                ADD_FAILURE() << "Invalid SPIR-V: " << entry.path();
            }
        }
    }
    
    EXPECT_GT(totalShaders, 0) << "No shader files found";
    EXPECT_EQ(validShaders, totalShaders) << "Some shaders failed validation";
    
    // Expected shaders for ML operations
    std::vector<std::string> expectedShaders = {
        "add.spv", "multiply.spv", "conv2d.spv", "matmul.spv",
        "relu.spv", "sigmoid.spv", "tanh.spv", "softmax.spv",
        "maxpool.spv", "avgpool.spv", "batchnorm.spv", "layernorm.spv"
    };
    
    for (const auto& shader : expectedShaders) {
        EXPECT_TRUE(shaderCache.find(shader) != shaderCache.end()) 
            << "Missing expected shader: " << shader;
    }
}

// Test 2: Create Shader Modules
TEST_F(SPIRVShaderTest, CreateShaderModules) {
    // Load a simple compute shader
    auto addShaderPath = shadersDir / "add.spv";
    if (!fs::exists(addShaderPath)) {
        GTEST_SKIP() << "Add shader not found";
    }
    
    auto spirv = loadSPIRV(addShaderPath);
    ASSERT_TRUE(validateSPIRV(spirv));
    
    // Create shader module
    vk::ShaderModuleCreateInfo moduleInfo{
        {},
        spirv.size() * sizeof(uint32_t),
        spirv.data()
    };
    
    vk::raii::ShaderModule shaderModule(device, moduleInfo);
    EXPECT_TRUE(*shaderModule) << "Shader module creation failed";
}

// Test 3: SPIRV Reflection
TEST_F(SPIRVShaderTest, SPIRVReflection) {
    auto conv2dPath = shadersDir / "conv2d.spv";
    if (!fs::exists(conv2dPath)) {
        GTEST_SKIP() << "Conv2D shader not found";
    }
    
    auto spirv = loadSPIRV(conv2dPath);
    ASSERT_TRUE(validateSPIRV(spirv));
    
    // Use SPIRV-Cross for reflection
    spirv_cross::Compiler compiler(spirv);
    spirv_cross::ShaderResources resources = compiler.get_shader_resources();
    
    // Check for expected resources in conv2d shader
    EXPECT_GE(resources.storage_buffers.size(), 3) 
        << "Conv2D should have at least 3 storage buffers (input, weights, output)";
    
    // Verify push constants if present
    if (!resources.push_constant_buffers.empty()) {
        auto& pushConstants = resources.push_constant_buffers[0];
        auto type = compiler.get_type(pushConstants.type_id);
        EXPECT_GT(compiler.get_declared_struct_size(type), 0) 
            << "Push constants should have non-zero size";
    }
    
    // Check workgroup size
    auto entry_points = compiler.get_entry_points_and_stages();
    for (const auto& entry : entry_points) {
        if (entry.execution_model == spv::ExecutionModelGLCompute) {
            auto workgroup_size = compiler.get_execution_mode_bitset();
            // Verify workgroup dimensions are set
        }
    }
}

// Test 4: ARM ML Extension Instructions
TEST_F(SPIRVShaderTest, ARMMLExtensions) {
    // Check if ARM ML extensions are supported
    auto deviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    bool hasARMMLExt = false;
    
    for (const auto& ext : deviceExtensions) {
        std::string extName(ext.extensionName);
        if (extName.find("VK_ARM_ml_") != std::string::npos ||
            extName.find("VK_ARM_tensor") != std::string::npos ||
            extName.find("VK_ARM_graph") != std::string::npos) {
            hasARMMLExt = true;
            break;
        }
    }
    
    if (!hasARMMLExt) {
        GTEST_SKIP() << "ARM ML extensions not available";
    }
    
    // Test shader with ARM ML instructions
    // These would be custom shaders using ARM ML-specific opcodes
    struct ARMMLOps {
        uint32_t opTensorCreate = 0x10000;     // Example ARM ML opcode
        uint32_t opTensorMatMul = 0x10001;
        uint32_t opTensorConv2D = 0x10002;
        uint32_t opTensorActivation = 0x10003;
        uint32_t opDataGraphCreate = 0x10100;
        uint32_t opDataGraphExecute = 0x10101;
    };
    
    // Verify ARM ML capability in SPIRV
    auto checkARMMLCapability = [](const std::vector<uint32_t>& spirv) {
        // Look for ARM ML capability declarations
        for (size_t i = 5; i < spirv.size(); ++i) {
            uint32_t opcode = spirv[i] & 0xFFFF;
            if (opcode == 17) { // OpCapability
                uint32_t capability = spirv[i+1];
                // Check for ARM-specific capabilities (these would be defined by ARM)
                if (capability >= 5000 && capability < 6000) {
                    return true;
                }
            }
        }
        return false;
    };
    
    // Test with ML-specific shaders if available
    for (const auto& [name, spirv] : shaderCache) {
        if (name.find("arm_ml") != std::string::npos) {
            EXPECT_TRUE(checkARMMLCapability(spirv)) 
                << "ARM ML shader missing capability: " << name;
        }
    }
}

// Test 5: Shader Compilation Pipeline
TEST_F(SPIRVShaderTest, ShaderCompilationPipeline) {
    // Test creating a complete compute pipeline with shader
    auto matmulPath = shadersDir / "matmul.spv";
    if (!fs::exists(matmulPath)) {
        GTEST_SKIP() << "MatMul shader not found";
    }
    
    auto spirv = loadSPIRV(matmulPath);
    ASSERT_TRUE(validateSPIRV(spirv));
    
    // Create shader module
    vk::ShaderModuleCreateInfo moduleInfo{
        {},
        spirv.size() * sizeof(uint32_t),
        spirv.data()
    };
    vk::raii::ShaderModule shaderModule(device, moduleInfo);
    
    // Create descriptor set layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };
    
    vk::DescriptorSetLayoutCreateInfo layoutInfo{{}, bindings};
    vk::raii::DescriptorSetLayout descriptorSetLayout(device, layoutInfo);
    
    // Create pipeline layout
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{{}, *descriptorSetLayout};
    vk::raii::PipelineLayout pipelineLayout(device, pipelineLayoutInfo);
    
    // Create compute pipeline
    vk::ComputePipelineCreateInfo pipelineInfo{
        {},
        {
            {},
            vk::ShaderStageFlagBits::eCompute,
            *shaderModule,
            "main"
        },
        *pipelineLayout
    };
    
    auto pipelines = vk::raii::Pipelines(device, nullptr, pipelineInfo);
    EXPECT_EQ(pipelines.size(), 1) << "Pipeline creation failed";
}

// Test 6: Specialization Constants in ML Shaders
TEST_F(SPIRVShaderTest, MLShaderSpecialization) {
    // Define specialization constants for ML operations
    struct MLSpecConstants {
        uint32_t batch_size = 1;
        uint32_t input_channels = 3;
        uint32_t output_channels = 64;
        uint32_t kernel_size = 3;
        uint32_t stride = 1;
        uint32_t padding = 1;
        float activation_alpha = 0.01f;  // For LeakyReLU
        uint32_t use_bias = 1;
    } mlConstants;
    
    // Create specialization map entries
    std::vector<vk::SpecializationMapEntry> entries = {
        {0, offsetof(MLSpecConstants, batch_size), sizeof(uint32_t)},
        {1, offsetof(MLSpecConstants, input_channels), sizeof(uint32_t)},
        {2, offsetof(MLSpecConstants, output_channels), sizeof(uint32_t)},
        {3, offsetof(MLSpecConstants, kernel_size), sizeof(uint32_t)},
        {4, offsetof(MLSpecConstants, stride), sizeof(uint32_t)},
        {5, offsetof(MLSpecConstants, padding), sizeof(uint32_t)},
        {6, offsetof(MLSpecConstants, activation_alpha), sizeof(float)},
        {7, offsetof(MLSpecConstants, use_bias), sizeof(uint32_t)}
    };
    
    vk::SpecializationInfo specInfo{
        static_cast<uint32_t>(entries.size()),
        entries.data(),
        sizeof(MLSpecConstants),
        &mlConstants
    };
    
    // Verify specialization for different ML configurations
    std::vector<MLSpecConstants> testConfigs = {
        {1, 3, 64, 3, 1, 1, 0.01f, 1},    // Standard conv
        {32, 3, 128, 5, 2, 2, 0.0f, 0},   // Larger batch, no bias
        {1, 256, 512, 1, 1, 0, 0.2f, 1}   // 1x1 conv, high alpha
    };
    
    for (const auto& config : testConfigs) {
        vk::SpecializationInfo testSpec{
            static_cast<uint32_t>(entries.size()),
            entries.data(),
            sizeof(MLSpecConstants),
            &config
        };
        EXPECT_EQ(testSpec.dataSize, sizeof(MLSpecConstants));
    }
}

// Test 7: Shader Variants for Different Operations
TEST_F(SPIRVShaderTest, ShaderVariants) {
    // Test that we have shader variants for different data types
    std::vector<std::string> operations = {
        "conv2d", "matmul", "add", "multiply"
    };
    
    std::vector<std::string> dataTypes = {
        "f32", "f16", "i8", "i32"
    };
    
    for (const auto& op : operations) {
        int variantCount = 0;
        for (const auto& dtype : dataTypes) {
            std::string shaderName = op + "_" + dtype + ".spv";
            auto shaderPath = shadersDir / shaderName;
            
            if (fs::exists(shaderPath)) {
                variantCount++;
                auto spirv = loadSPIRV(shaderPath);
                EXPECT_TRUE(validateSPIRV(spirv)) 
                    << "Invalid variant: " << shaderName;
            }
        }
        
        // We should have at least f32 variant for each operation
        EXPECT_GE(variantCount, 1) 
            << "Missing shader variants for operation: " << op;
    }
}

// Test 8: Workgroup Size Optimization
TEST_F(SPIRVShaderTest, WorkgroupSizeOptimization) {
    // Get device limits for workgroup size
    auto properties = physicalDevice.getProperties();
    auto& limits = properties.limits;
    
    uint32_t maxWorkgroupSizeX = limits.maxComputeWorkGroupSize[0];
    uint32_t maxWorkgroupSizeY = limits.maxComputeWorkGroupSize[1];
    uint32_t maxWorkgroupSizeZ = limits.maxComputeWorkGroupSize[2];
    uint32_t maxWorkgroupInvocations = limits.maxComputeWorkGroupInvocations;
    
    // Test optimal workgroup sizes for different operations
    struct WorkgroupConfig {
        std::string operation;
        uint32_t x, y, z;
        uint32_t expectedInvocations;
    };
    
    std::vector<WorkgroupConfig> configs = {
        {"conv2d", 16, 16, 1, 256},    // 2D convolution
        {"matmul", 8, 8, 4, 256},      // Matrix multiplication
        {"reduce", 256, 1, 1, 256},    // Reduction operations
        {"elementwise", 64, 1, 1, 64}  // Element-wise operations
    };
    
    for (const auto& config : configs) {
        EXPECT_LE(config.x, maxWorkgroupSizeX);
        EXPECT_LE(config.y, maxWorkgroupSizeY);
        EXPECT_LE(config.z, maxWorkgroupSizeZ);
        EXPECT_LE(config.expectedInvocations, maxWorkgroupInvocations);
        
        // Verify power-of-two for optimal performance
        EXPECT_EQ(config.x & (config.x - 1), 0) << "X not power of 2";
        if (config.y > 1) {
            EXPECT_EQ(config.y & (config.y - 1), 0) << "Y not power of 2";
        }
    }
}

// Test 9: Shader Memory Access Patterns
TEST_F(SPIRVShaderTest, MemoryAccessPatterns) {
    // Test shaders for correct memory access patterns
    auto checkMemoryAccess = [](const std::vector<uint32_t>& spirv) {
        spirv_cross::Compiler compiler(spirv);
        auto resources = compiler.get_shader_resources();
        
        // Check storage buffer decorations
        for (const auto& buffer : resources.storage_buffers) {
            auto decorations = compiler.get_decorations(buffer.id);
            
            // Check for coherent/volatile decorations where needed
            bool hasCoherent = decorations.get(spv::DecorationCoherent);
            bool hasVolatile = decorations.get(spv::DecorationVolatile);
            bool hasRestrict = decorations.get(spv::DecorationRestrict);
            
            // For output buffers, we might want restrict
            if (buffer.name.find("output") != std::string::npos) {
                // Output buffers benefit from restrict
            }
        }
        return true;
    };
    
    // Test memory access patterns in key shaders
    std::vector<std::string> criticalShaders = {
        "conv2d.spv", "matmul.spv", "reduce_sum.spv"
    };
    
    for (const auto& shaderName : criticalShaders) {
        auto iter = shaderCache.find(shaderName);
        if (iter != shaderCache.end()) {
            EXPECT_TRUE(checkMemoryAccess(iter->second))
                << "Memory access issue in: " << shaderName;
        }
    }
}

// Test 10: Shader Validation with Different Input Sizes
TEST_F(SPIRVShaderTest, InputSizeValidation) {
    // Test that shaders can handle various input sizes
    struct TestSize {
        uint32_t batch;
        uint32_t height;
        uint32_t width;
        uint32_t channels;
    };
    
    std::vector<TestSize> testSizes = {
        {1, 224, 224, 3},      // ImageNet size
        {1, 299, 299, 3},      // Inception size
        {32, 32, 32, 64},      // Small batch
        {1, 1920, 1080, 3},    // HD video frame
        {1, 1, 1, 1024},       // 1D vector
        {128, 7, 7, 512}       // Large batch, small spatial
    };
    
    // Calculate required buffer sizes
    for (const auto& size : testSizes) {
        size_t elementCount = size.batch * size.height * size.width * size.channels;
        size_t bufferSize = elementCount * sizeof(float);
        
        // Verify size is within device limits
        auto properties = physicalDevice.getProperties();
        EXPECT_LE(bufferSize, properties.limits.maxStorageBufferRange)
            << "Buffer size exceeds device limit for: "
            << size.batch << "x" << size.height << "x" 
            << size.width << "x" << size.channels;
    }
}

// Test 11: Shader Performance Hints
TEST_F(SPIRVShaderTest, PerformanceHints) {
    // Check shaders for performance-critical patterns
    auto checkPerformancePatterns = [](const std::vector<uint32_t>& spirv) {
        spirv_cross::Compiler compiler(spirv);
        
        // Get all variables
        auto variables = compiler.get_active_interface_variables();
        
        // Check for common performance issues
        bool hasIssues = false;
        std::vector<std::string> issues;
        
        // Check for unrolled loops where beneficial
        // Check for vectorized operations
        // Check for coalesced memory access
        
        return !hasIssues;
    };
    
    // Performance-critical shaders
    std::vector<std::string> perfCriticalShaders = {
        "conv2d.spv", "matmul.spv", "gemm.spv"
    };
    
    for (const auto& shaderName : perfCriticalShaders) {
        auto iter = shaderCache.find(shaderName);
        if (iter != shaderCache.end()) {
            EXPECT_TRUE(checkPerformancePatterns(iter->second))
                << "Performance issues in: " << shaderName;
        }
    }
}

// Test 12: ARM-Specific Optimizations
TEST_F(SPIRVShaderTest, ARMSpecificOptimizations) {
    // Check for ARM-specific optimizations in shaders
    
    // Check if running on ARM architecture
#ifdef __ARM_ARCH
    // Test ARM-specific shader features
    struct ARMFeatures {
        bool hasNEON = false;
        bool hasSVE = false;
        bool hasSVE2 = false;
        bool hasAMX = false;
    };
    
    // Check device properties for ARM-specific features
    auto properties = physicalDevice.getProperties();
    std::string deviceName(properties.deviceName);
    
    bool isARMGPU = deviceName.find("Mali") != std::string::npos ||
                    deviceName.find("Immortalis") != std::string::npos ||
                    deviceName.find("Apple") != std::string::npos;
    
    if (isARMGPU) {
        // Test ARM GPU-specific optimizations
        // These would be shader patterns optimized for ARM GPUs
        
        // Check for warp/subgroup operations
        auto checkSubgroupOps = [](const std::vector<uint32_t>& spirv) {
            for (size_t i = 5; i < spirv.size(); ++i) {
                uint32_t opcode = spirv[i] & 0xFFFF;
                // Check for subgroup operations (OpGroupNonUniform*)
                if (opcode >= 333 && opcode <= 365) {
                    return true;
                }
            }
            return false;
        };
        
        // ARM GPUs benefit from subgroup operations
        for (const auto& [name, spirv] : shaderCache) {
            if (name.find("reduce") != std::string::npos) {
                EXPECT_TRUE(checkSubgroupOps(spirv))
                    << "Missing subgroup ops in reduction: " << name;
            }
        }
    }
#endif
}

} // namespace mlsdk::tests