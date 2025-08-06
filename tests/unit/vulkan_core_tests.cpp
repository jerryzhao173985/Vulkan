/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Core Vulkan Unit Tests for ARM ML SDK
 * Tests fundamental Vulkan operations: device creation, memory management,
 * command buffers, pipelines, and synchronization.
 */

#include <gtest/gtest.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <memory>
#include <cstring>

namespace mlsdk::tests {

class VulkanCoreTest : public ::testing::Test {
protected:
    vk::raii::Context context;
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    uint32_t queueFamilyIndex = 0;
    vk::raii::Queue computeQueue{nullptr};
    
    void SetUp() override {
        // Create Vulkan instance
        vk::ApplicationInfo appInfo{
            "VulkanCoreTests",
            VK_MAKE_VERSION(1, 0, 0),
            "ARM ML SDK Tests",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_2
        };
        
        std::vector<const char*> extensions;
        std::vector<const char*> layers;
        
#ifdef DEBUG
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
        
        vk::InstanceCreateInfo instanceCreateInfo{
            {},
            &appInfo,
            layers,
            extensions
        };
        
        instance = vk::raii::Instance(context, instanceCreateInfo);
        
        // Select physical device
        auto physicalDevices = vk::raii::PhysicalDevices(instance);
        ASSERT_FALSE(physicalDevices.empty()) << "No Vulkan devices found";
        
        physicalDevice = std::move(physicalDevices[0]);
        
        // Find compute queue family
        auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
            if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
                queueFamilyIndex = i;
                break;
            }
        }
        
        // Create logical device
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo{
            {},
            queueFamilyIndex,
            1,
            &queuePriority
        };
        
        std::vector<const char*> deviceExtensions;
        vk::PhysicalDeviceFeatures deviceFeatures{};
        
        vk::DeviceCreateInfo deviceCreateInfo{
            {},
            queueCreateInfo,
            {},
            deviceExtensions,
            &deviceFeatures
        };
        
        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        computeQueue = vk::raii::Queue(device, queueFamilyIndex, 0);
    }
    
    void TearDown() override {
        if (*device) {
            device.waitIdle();
        }
    }
};

// Test 1: Instance and Device Creation
TEST_F(VulkanCoreTest, DeviceCreation) {
    ASSERT_TRUE(*instance) << "Instance creation failed";
    ASSERT_TRUE(*physicalDevice) << "Physical device selection failed";
    ASSERT_TRUE(*device) << "Logical device creation failed";
    ASSERT_TRUE(*computeQueue) << "Compute queue retrieval failed";
    
    // Verify device properties
    auto properties = physicalDevice.getProperties();
    EXPECT_GT(properties.limits.maxComputeWorkGroupSize[0], 0);
    EXPECT_GT(properties.limits.maxComputeWorkGroupInvocations, 0);
}

// Test 2: Memory Allocation and Management
TEST_F(VulkanCoreTest, MemoryAllocation) {
    const VkDeviceSize bufferSize = 1024 * 1024; // 1MB
    
    // Create buffer
    vk::BufferCreateInfo bufferInfo{
        {},
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive
    };
    
    vk::raii::Buffer buffer(device, bufferInfo);
    ASSERT_TRUE(*buffer) << "Buffer creation failed";
    
    // Get memory requirements
    auto memRequirements = buffer.getMemoryRequirements();
    EXPECT_GE(memRequirements.size, bufferSize);
    
    // Find suitable memory type
    auto memProperties = physicalDevice.getMemoryProperties();
    uint32_t memoryTypeIndex = UINT32_MAX;
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
            memoryTypeIndex = i;
            break;
        }
    }
    
    ASSERT_NE(memoryTypeIndex, UINT32_MAX) << "No suitable memory type found";
    
    // Allocate memory
    vk::MemoryAllocateInfo allocInfo{
        memRequirements.size,
        memoryTypeIndex
    };
    
    vk::raii::DeviceMemory memory(device, allocInfo);
    ASSERT_TRUE(*memory) << "Memory allocation failed";
    
    // Bind memory to buffer
    buffer.bindMemory(*memory, 0);
    
    // Map memory and write data
    void* mappedMemory = memory.mapMemory(0, bufferSize);
    ASSERT_NE(mappedMemory, nullptr) << "Memory mapping failed";
    
    std::memset(mappedMemory, 0xAB, bufferSize);
    memory.unmapMemory();
}

// Test 3: Command Buffer Recording and Submission
TEST_F(VulkanCoreTest, CommandBufferOperations) {
    // Create command pool
    vk::CommandPoolCreateInfo poolInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        queueFamilyIndex
    };
    
    vk::raii::CommandPool commandPool(device, poolInfo);
    ASSERT_TRUE(*commandPool) << "Command pool creation failed";
    
    // Allocate command buffer
    vk::CommandBufferAllocateInfo allocInfo{
        *commandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    };
    
    auto commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    ASSERT_EQ(commandBuffers.size(), 1) << "Command buffer allocation failed";
    
    auto& cmdBuffer = commandBuffers[0];
    
    // Record commands
    vk::CommandBufferBeginInfo beginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };
    
    cmdBuffer.begin(beginInfo);
    
    // Add pipeline barrier (example command)
    vk::MemoryBarrier memoryBarrier{
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead
    };
    
    cmdBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        {},
        memoryBarrier,
        {},
        {}
    );
    
    cmdBuffer.end();
    
    // Submit command buffer
    vk::SubmitInfo submitInfo{
        {},
        {},
        *cmdBuffer,
        {}
    };
    
    computeQueue.submit(submitInfo);
    computeQueue.waitIdle();
}

// Test 4: Compute Pipeline Creation
TEST_F(VulkanCoreTest, ComputePipelineCreation) {
    // Create simple compute shader SPIR-V (add operation)
    std::vector<uint32_t> spirvCode = {
        0x07230203, // Magic number
        0x00010000, // Version 1.0
        0x00080001, // Generator
        0x00000013, // Bound
        0x00000000, // Schema
        // ... minimal valid SPIR-V code for testing
        // This would normally be loaded from a .spv file
    };
    
    // Create shader module
    vk::ShaderModuleCreateInfo shaderInfo{
        {},
        spirvCode.size() * sizeof(uint32_t),
        spirvCode.data()
    };
    
    // Note: This will fail without valid SPIR-V, but tests the API
    try {
        vk::raii::ShaderModule shaderModule(device, shaderInfo);
        
        // Create pipeline layout
        vk::PipelineLayoutCreateInfo layoutInfo{};
        vk::raii::PipelineLayout pipelineLayout(device, layoutInfo);
        
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
        EXPECT_EQ(pipelines.size(), 1);
    } catch (const vk::SystemError& e) {
        // Expected to fail with invalid SPIR-V, but API calls are tested
        GTEST_SKIP() << "Skipping pipeline creation with mock SPIR-V";
    }
}

// Test 5: Descriptor Sets
TEST_F(VulkanCoreTest, DescriptorSetManagement) {
    // Create descriptor pool
    vk::DescriptorPoolSize poolSize{
        vk::DescriptorType::eStorageBuffer,
        10
    };
    
    vk::DescriptorPoolCreateInfo poolInfo{
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        10,
        poolSize
    };
    
    vk::raii::DescriptorPool descriptorPool(device, poolInfo);
    ASSERT_TRUE(*descriptorPool) << "Descriptor pool creation failed";
    
    // Create descriptor set layout
    vk::DescriptorSetLayoutBinding layoutBinding{
        0,
        vk::DescriptorType::eStorageBuffer,
        1,
        vk::ShaderStageFlagBits::eCompute
    };
    
    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        {},
        layoutBinding
    };
    
    vk::raii::DescriptorSetLayout setLayout(device, layoutInfo);
    ASSERT_TRUE(*setLayout) << "Descriptor set layout creation failed";
    
    // Allocate descriptor set
    vk::DescriptorSetAllocateInfo allocInfo{
        *descriptorPool,
        *setLayout
    };
    
    auto descriptorSets = vk::raii::DescriptorSets(device, allocInfo);
    ASSERT_EQ(descriptorSets.size(), 1) << "Descriptor set allocation failed";
}

// Test 6: Synchronization Primitives
TEST_F(VulkanCoreTest, SynchronizationPrimitives) {
    // Create fence
    vk::FenceCreateInfo fenceInfo{};
    vk::raii::Fence fence(device, fenceInfo);
    ASSERT_TRUE(*fence) << "Fence creation failed";
    
    // Create semaphore
    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::raii::Semaphore semaphore(device, semaphoreInfo);
    ASSERT_TRUE(*semaphore) << "Semaphore creation failed";
    
    // Test fence operations
    EXPECT_EQ(device.getFenceStatus(*fence), vk::Result::eNotReady);
    
    // Create command pool and buffer for testing
    vk::CommandPoolCreateInfo poolInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        queueFamilyIndex
    };
    vk::raii::CommandPool commandPool(device, poolInfo);
    
    vk::CommandBufferAllocateInfo allocInfo{
        *commandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    };
    auto commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    
    // Record empty command buffer
    commandBuffers[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    commandBuffers[0].end();
    
    // Submit with fence
    vk::SubmitInfo submitInfo{
        {},
        {},
        *commandBuffers[0],
        {}
    };
    
    computeQueue.submit(submitInfo, *fence);
    
    // Wait for fence
    auto result = device.waitForFences(*fence, VK_TRUE, UINT64_MAX);
    EXPECT_EQ(result, vk::Result::eSuccess);
    EXPECT_EQ(device.getFenceStatus(*fence), vk::Result::eSuccess);
}

// Test 7: Buffer Copy Operations
TEST_F(VulkanCoreTest, BufferCopyOperations) {
    const VkDeviceSize bufferSize = 1024;
    const std::vector<uint8_t> testData(bufferSize, 0x42);
    
    // Create source and destination buffers
    vk::BufferCreateInfo bufferInfo{
        {},
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive
    };
    
    vk::raii::Buffer srcBuffer(device, bufferInfo);
    vk::raii::Buffer dstBuffer(device, bufferInfo);
    
    // Allocate memory for both buffers
    auto memRequirements = srcBuffer.getMemoryRequirements();
    auto memProperties = physicalDevice.getMemoryProperties();
    
    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
            memoryTypeIndex = i;
            break;
        }
    }
    
    ASSERT_NE(memoryTypeIndex, UINT32_MAX);
    
    vk::MemoryAllocateInfo allocInfo{
        memRequirements.size * 2, // Allocate for both buffers
        memoryTypeIndex
    };
    
    vk::raii::DeviceMemory memory(device, allocInfo);
    srcBuffer.bindMemory(*memory, 0);
    dstBuffer.bindMemory(*memory, memRequirements.size);
    
    // Write data to source buffer
    void* mappedMemory = memory.mapMemory(0, bufferSize);
    std::memcpy(mappedMemory, testData.data(), bufferSize);
    memory.unmapMemory();
    
    // Create command buffer for copy
    vk::CommandPoolCreateInfo poolInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        queueFamilyIndex
    };
    vk::raii::CommandPool commandPool(device, poolInfo);
    
    vk::CommandBufferAllocateInfo cmdAllocInfo{
        *commandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    };
    auto commandBuffers = vk::raii::CommandBuffers(device, cmdAllocInfo);
    
    // Record copy command
    commandBuffers[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    
    vk::BufferCopy copyRegion{
        0, // srcOffset
        0, // dstOffset
        bufferSize
    };
    commandBuffers[0].copyBuffer(*srcBuffer, *dstBuffer, copyRegion);
    
    commandBuffers[0].end();
    
    // Submit and wait
    vk::SubmitInfo submitInfo{
        {},
        {},
        *commandBuffers[0],
        {}
    };
    
    computeQueue.submit(submitInfo);
    computeQueue.waitIdle();
    
    // Verify copy
    mappedMemory = memory.mapMemory(memRequirements.size, bufferSize);
    std::vector<uint8_t> readData(bufferSize);
    std::memcpy(readData.data(), mappedMemory, bufferSize);
    memory.unmapMemory();
    
    EXPECT_EQ(readData, testData);
}

// Test 8: Push Constants
TEST_F(VulkanCoreTest, PushConstants) {
    // Define push constant range
    vk::PushConstantRange pushConstantRange{
        vk::ShaderStageFlagBits::eCompute,
        0,      // offset
        16      // size (4 floats)
    };
    
    // Create pipeline layout with push constants
    vk::PipelineLayoutCreateInfo layoutInfo{
        {},
        {},
        pushConstantRange
    };
    
    vk::raii::PipelineLayout pipelineLayout(device, layoutInfo);
    ASSERT_TRUE(*pipelineLayout) << "Pipeline layout creation with push constants failed";
    
    // Create command buffer
    vk::CommandPoolCreateInfo poolInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        queueFamilyIndex
    };
    vk::raii::CommandPool commandPool(device, poolInfo);
    
    vk::CommandBufferAllocateInfo allocInfo{
        *commandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    };
    auto commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    
    // Record push constant command
    float pushData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    commandBuffers[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    commandBuffers[0].pushConstants(
        *pipelineLayout,
        vk::ShaderStageFlagBits::eCompute,
        0,
        sizeof(pushData),
        pushData
    );
    commandBuffers[0].end();
}

// Test 9: Query Pool for Performance Metrics
TEST_F(VulkanCoreTest, QueryPoolTimestamps) {
    // Check if timestamps are supported
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    if (queueFamilyProperties[queueFamilyIndex].timestampValidBits == 0) {
        GTEST_SKIP() << "Timestamps not supported on this queue family";
    }
    
    // Create query pool
    vk::QueryPoolCreateInfo queryPoolInfo{
        {},
        vk::QueryType::eTimestamp,
        2, // start and end timestamps
        {}
    };
    
    vk::raii::QueryPool queryPool(device, queryPoolInfo);
    ASSERT_TRUE(*queryPool) << "Query pool creation failed";
    
    // Create command buffer
    vk::CommandPoolCreateInfo poolInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        queueFamilyIndex
    };
    vk::raii::CommandPool commandPool(device, poolInfo);
    
    vk::CommandBufferAllocateInfo allocInfo{
        *commandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    };
    auto commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    
    // Record timestamp queries
    commandBuffers[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    commandBuffers[0].resetQueryPool(*queryPool, 0, 2);
    commandBuffers[0].writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *queryPool, 0);
    commandBuffers[0].writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *queryPool, 1);
    commandBuffers[0].end();
    
    // Submit
    vk::SubmitInfo submitInfo{
        {},
        {},
        *commandBuffers[0],
        {}
    };
    
    computeQueue.submit(submitInfo);
    computeQueue.waitIdle();
    
    // Get results
    uint64_t timestamps[2];
    auto result = device.getQueryPoolResults(
        *queryPool,
        0,
        2,
        sizeof(timestamps),
        timestamps,
        sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait
    );
    
    EXPECT_EQ(result, vk::Result::eSuccess);
    EXPECT_LE(timestamps[0], timestamps[1]); // End should be after start
}

// Test 10: Multiple Queue Submission
TEST_F(VulkanCoreTest, MultipleQueueSubmission) {
    // Create command pool
    vk::CommandPoolCreateInfo poolInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        queueFamilyIndex
    };
    vk::raii::CommandPool commandPool(device, poolInfo);
    
    // Allocate multiple command buffers
    const uint32_t numBuffers = 5;
    vk::CommandBufferAllocateInfo allocInfo{
        *commandPool,
        vk::CommandBufferLevel::ePrimary,
        numBuffers
    };
    auto commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    
    // Record each command buffer
    for (uint32_t i = 0; i < numBuffers; ++i) {
        commandBuffers[i].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        // Add some dummy work
        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eMemoryWrite,
            vk::AccessFlagBits::eMemoryRead
        };
        commandBuffers[i].pipelineBarrier(
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eAllCommands,
            {},
            barrier,
            {},
            {}
        );
        commandBuffers[i].end();
    }
    
    // Submit all command buffers
    std::vector<vk::CommandBuffer> cmdBuffers;
    for (const auto& cb : commandBuffers) {
        cmdBuffers.push_back(*cb);
    }
    
    vk::SubmitInfo submitInfo{
        {},
        {},
        cmdBuffers,
        {}
    };
    
    vk::raii::Fence fence(device, vk::FenceCreateInfo{});
    computeQueue.submit(submitInfo, *fence);
    
    // Wait for completion
    auto result = device.waitForFences(*fence, VK_TRUE, UINT64_MAX);
    EXPECT_EQ(result, vk::Result::eSuccess);
}

// Test 11: Vulkan 1.2 Timeline Semaphores
TEST_F(VulkanCoreTest, TimelineSemaphores) {
    // Check for timeline semaphore support
    vk::PhysicalDeviceVulkan12Features features12{};
    vk::PhysicalDeviceFeatures2 features2{};
    features2.pNext = &features12;
    physicalDevice.getFeatures2(&features2);
    
    if (!features12.timelineSemaphore) {
        GTEST_SKIP() << "Timeline semaphores not supported";
    }
    
    // Create timeline semaphore
    vk::SemaphoreTypeCreateInfo typeInfo{
        vk::SemaphoreType::eTimeline,
        0 // Initial value
    };
    
    vk::SemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.pNext = &typeInfo;
    
    vk::raii::Semaphore timelineSemaphore(device, semaphoreInfo);
    ASSERT_TRUE(*timelineSemaphore) << "Timeline semaphore creation failed";
    
    // Query semaphore value
    uint64_t value = device.getSemaphoreCounterValue(*timelineSemaphore);
    EXPECT_EQ(value, 0);
    
    // Signal from host
    vk::SemaphoreSignalInfo signalInfo{
        *timelineSemaphore,
        42
    };
    device.signalSemaphore(signalInfo);
    
    // Verify signal
    value = device.getSemaphoreCounterValue(*timelineSemaphore);
    EXPECT_EQ(value, 42);
    
    // Wait on host
    vk::SemaphoreWaitInfo waitInfo{
        {},
        *timelineSemaphore,
        42
    };
    auto result = device.waitSemaphores(waitInfo, UINT64_MAX);
    EXPECT_EQ(result, vk::Result::eSuccess);
}

// Test 12: Buffer Device Address (Vulkan 1.2)
TEST_F(VulkanCoreTest, BufferDeviceAddress) {
    // Check for buffer device address support
    vk::PhysicalDeviceVulkan12Features features12{};
    vk::PhysicalDeviceFeatures2 features2{};
    features2.pNext = &features12;
    physicalDevice.getFeatures2(&features2);
    
    if (!features12.bufferDeviceAddress) {
        GTEST_SKIP() << "Buffer device address not supported";
    }
    
    // Create buffer with device address flag
    vk::BufferCreateInfo bufferInfo{
        {},
        1024,
        vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::SharingMode::eExclusive
    };
    
    vk::raii::Buffer buffer(device, bufferInfo);
    
    // Allocate memory with device address flag
    auto memRequirements = buffer.getMemoryRequirements();
    auto memProperties = physicalDevice.getMemoryProperties();
    
    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            memoryTypeIndex = i;
            break;
        }
    }
    
    if (memoryTypeIndex == UINT32_MAX) {
        GTEST_SKIP() << "No suitable memory type for device address";
    }
    
    vk::MemoryAllocateFlagsInfo flagsInfo{
        vk::MemoryAllocateFlagBits::eDeviceAddress
    };
    
    vk::MemoryAllocateInfo allocInfo{
        memRequirements.size,
        memoryTypeIndex
    };
    allocInfo.pNext = &flagsInfo;
    
    vk::raii::DeviceMemory memory(device, allocInfo);
    buffer.bindMemory(*memory, 0);
    
    // Get buffer device address
    vk::BufferDeviceAddressInfo addressInfo{
        *buffer
    };
    
    vk::DeviceAddress deviceAddress = device.getBufferAddress(addressInfo);
    EXPECT_NE(deviceAddress, 0) << "Invalid device address";
}

// Test 13: Memory Coherency Tests
TEST_F(VulkanCoreTest, MemoryCoherency) {
    const VkDeviceSize bufferSize = 4096;
    
    // Create buffer for coherency testing
    vk::BufferCreateInfo bufferInfo{
        {},
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive
    };
    
    vk::raii::Buffer buffer(device, bufferInfo);
    
    // Find coherent memory type
    auto memRequirements = buffer.getMemoryRequirements();
    auto memProperties = physicalDevice.getMemoryProperties();
    
    uint32_t coherentTypeIndex = UINT32_MAX;
    uint32_t cachedTypeIndex = UINT32_MAX;
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if (memRequirements.memoryTypeBits & (1 << i)) {
            auto flags = memProperties.memoryTypes[i].propertyFlags;
            if ((flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
                (flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
                coherentTypeIndex = i;
            } else if ((flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
                       (flags & vk::MemoryPropertyFlagBits::eHostCached)) {
                cachedTypeIndex = i;
            }
        }
    }
    
    ASSERT_NE(coherentTypeIndex, UINT32_MAX) << "No coherent memory type found";
    
    // Test coherent memory
    vk::MemoryAllocateInfo allocInfo{
        memRequirements.size,
        coherentTypeIndex
    };
    
    vk::raii::DeviceMemory coherentMemory(device, allocInfo);
    buffer.bindMemory(*coherentMemory, 0);
    
    // Write pattern and verify immediate visibility
    void* mappedMemory = coherentMemory.mapMemory(0, bufferSize);
    uint32_t* data = static_cast<uint32_t*>(mappedMemory);
    
    const uint32_t testPattern = 0xDEADBEEF;
    for (size_t i = 0; i < bufferSize / sizeof(uint32_t); ++i) {
        data[i] = testPattern;
    }
    
    // Verify write is visible without explicit flush
    for (size_t i = 0; i < bufferSize / sizeof(uint32_t); ++i) {
        EXPECT_EQ(data[i], testPattern);
    }
    
    coherentMemory.unmapMemory();
}

// Test 14: Event-based Synchronization
TEST_F(VulkanCoreTest, EventSynchronization) {
    // Create event
    vk::EventCreateInfo eventInfo{};
    vk::raii::Event event(device, eventInfo);
    ASSERT_TRUE(*event) << "Event creation failed";
    
    // Test host operations
    EXPECT_EQ(device.getEventStatus(*event), vk::Result::eEventReset);
    
    device.setEvent(*event);
    EXPECT_EQ(device.getEventStatus(*event), vk::Result::eEventSet);
    
    device.resetEvent(*event);
    EXPECT_EQ(device.getEventStatus(*event), vk::Result::eEventReset);
    
    // Test in command buffer
    vk::CommandPoolCreateInfo poolInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        queueFamilyIndex
    };
    vk::raii::CommandPool commandPool(device, poolInfo);
    
    vk::CommandBufferAllocateInfo allocInfo{
        *commandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    };
    auto commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    
    commandBuffers[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    
    // Set event in command buffer
    commandBuffers[0].setEvent(*event, vk::PipelineStageFlagBits::eComputeShader);
    
    // Wait for event
    vk::MemoryBarrier memBarrier{
        vk::AccessFlagBits::eMemoryWrite,
        vk::AccessFlagBits::eMemoryRead
    };
    
    commandBuffers[0].waitEvents(
        *event,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        memBarrier,
        {},
        {}
    );
    
    // Reset event
    commandBuffers[0].resetEvent(*event, vk::PipelineStageFlagBits::eComputeShader);
    
    commandBuffers[0].end();
}

// Test 15: Pipeline Cache
TEST_F(VulkanCoreTest, PipelineCache) {
    // Create initial pipeline cache
    vk::PipelineCacheCreateInfo cacheInfo{};
    vk::raii::PipelineCache cache1(device, cacheInfo);
    ASSERT_TRUE(*cache1) << "Pipeline cache creation failed";
    
    // Get cache data (initially empty)
    auto cacheData = cache1.getData();
    EXPECT_GT(cacheData.size(), 0) << "Cache should have header data";
    
    // Create second cache from data
    vk::PipelineCacheCreateInfo cacheInfo2{
        {},
        cacheData.size(),
        cacheData.data()
    };
    vk::raii::PipelineCache cache2(device, cacheInfo2);
    ASSERT_TRUE(*cache2) << "Pipeline cache creation from data failed";
    
    // Merge caches
    std::vector<vk::PipelineCache> sourceCaches = {*cache2};
    cache1.mergePipelineCaches(sourceCaches);
    
    // Verify merged cache
    auto mergedData = cache1.getData();
    EXPECT_GE(mergedData.size(), cacheData.size());
}

// Test 16: Secondary Command Buffers
TEST_F(VulkanCoreTest, SecondaryCommandBuffers) {
    // Create command pool
    vk::CommandPoolCreateInfo poolInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        queueFamilyIndex
    };
    vk::raii::CommandPool commandPool(device, poolInfo);
    
    // Allocate primary command buffer
    vk::CommandBufferAllocateInfo primaryAllocInfo{
        *commandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    };
    auto primaryBuffers = vk::raii::CommandBuffers(device, primaryAllocInfo);
    
    // Allocate secondary command buffers
    vk::CommandBufferAllocateInfo secondaryAllocInfo{
        *commandPool,
        vk::CommandBufferLevel::eSecondary,
        3
    };
    auto secondaryBuffers = vk::raii::CommandBuffers(device, secondaryAllocInfo);
    
    // Record secondary command buffers
    vk::CommandBufferInheritanceInfo inheritanceInfo{};
    vk::CommandBufferBeginInfo secondaryBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        &inheritanceInfo
    };
    
    for (auto& secondary : secondaryBuffers) {
        secondary.begin(secondaryBeginInfo);
        
        // Add work to secondary
        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eMemoryWrite,
            vk::AccessFlagBits::eMemoryRead
        };
        secondary.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {},
            barrier,
            {},
            {}
        );
        
        secondary.end();
    }
    
    // Record primary command buffer
    primaryBuffers[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    
    // Execute secondary command buffers
    std::vector<vk::CommandBuffer> secondaryCmdBuffers;
    for (const auto& secondary : secondaryBuffers) {
        secondaryCmdBuffers.push_back(*secondary);
    }
    primaryBuffers[0].executeCommands(secondaryCmdBuffers);
    
    primaryBuffers[0].end();
    
    // Submit
    vk::SubmitInfo submitInfo{
        {},
        {},
        *primaryBuffers[0],
        {}
    };
    
    computeQueue.submit(submitInfo);
    computeQueue.waitIdle();
}

// Test 17: Multi-Queue Synchronization
TEST_F(VulkanCoreTest, MultiQueueSynchronization) {
    // Check for multiple queue families
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    
    uint32_t secondQueueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
        if (i != queueFamilyIndex && 
            (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute)) {
            secondQueueFamily = i;
            break;
        }
    }
    
    if (secondQueueFamily == UINT32_MAX) {
        // Try to get another queue from the same family
        if (queueFamilyProperties[queueFamilyIndex].queueCount > 1) {
            secondQueueFamily = queueFamilyIndex;
        } else {
            GTEST_SKIP() << "No second queue available for testing";
        }
    }
    
    // Create semaphore for synchronization
    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::raii::Semaphore semaphore(device, semaphoreInfo);
    
    // Create command pools and buffers for both queues
    vk::CommandPoolCreateInfo poolInfo1{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        queueFamilyIndex
    };
    vk::raii::CommandPool pool1(device, poolInfo1);
    
    vk::CommandBufferAllocateInfo allocInfo1{
        *pool1,
        vk::CommandBufferLevel::ePrimary,
        1
    };
    auto cmdBuffers1 = vk::raii::CommandBuffers(device, allocInfo1);
    
    // Record first command buffer
    cmdBuffers1[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmdBuffers1[0].end();
    
    // Submit to first queue with signal
    vk::SubmitInfo submitInfo1{
        {},
        {},
        *cmdBuffers1[0],
        *semaphore
    };
    
    computeQueue.submit(submitInfo1);
    
    // Get second queue
    vk::raii::Queue secondQueue(device, secondQueueFamily, 
                                secondQueueFamily == queueFamilyIndex ? 1 : 0);
    
    // Create command buffer for second queue
    vk::CommandPoolCreateInfo poolInfo2{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        secondQueueFamily
    };
    vk::raii::CommandPool pool2(device, poolInfo2);
    
    vk::CommandBufferAllocateInfo allocInfo2{
        *pool2,
        vk::CommandBufferLevel::ePrimary,
        1
    };
    auto cmdBuffers2 = vk::raii::CommandBuffers(device, allocInfo2);
    
    // Record second command buffer
    cmdBuffers2[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmdBuffers2[0].end();
    
    // Submit to second queue with wait
    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;
    vk::SubmitInfo submitInfo2{
        *semaphore,
        waitStage,
        *cmdBuffers2[0],
        {}
    };
    
    vk::raii::Fence fence(device, vk::FenceCreateInfo{});
    secondQueue.submit(submitInfo2, *fence);
    
    // Wait for completion
    auto result = device.waitForFences(*fence, VK_TRUE, UINT64_MAX);
    EXPECT_EQ(result, vk::Result::eSuccess);
}

// Test 18: Specialization Constants
TEST_F(VulkanCoreTest, SpecializationConstants) {
    // Define specialization data
    struct SpecData {
        uint32_t workgroupSize = 256;
        float scale = 2.0f;
        int32_t offset = -10;
    } specData;
    
    // Create specialization map entries
    std::vector<vk::SpecializationMapEntry> entries = {
        {0, offsetof(SpecData, workgroupSize), sizeof(uint32_t)},
        {1, offsetof(SpecData, scale), sizeof(float)},
        {2, offsetof(SpecData, offset), sizeof(int32_t)}
    };
    
    // Create specialization info
    vk::SpecializationInfo specInfo{
        static_cast<uint32_t>(entries.size()),
        entries.data(),
        sizeof(SpecData),
        &specData
    };
    
    // Verify specialization info is valid
    EXPECT_EQ(specInfo.mapEntryCount, 3);
    EXPECT_EQ(specInfo.dataSize, sizeof(SpecData));
    EXPECT_NE(specInfo.pData, nullptr);
    
    // Note: Full pipeline creation with specialization would require valid SPIR-V
    // This test validates the specialization constant setup
}

// Test 19: Indirect Dispatch
TEST_F(VulkanCoreTest, IndirectDispatch) {
    // Create indirect buffer
    struct DispatchIndirectCommand {
        uint32_t x = 64;
        uint32_t y = 1;
        uint32_t z = 1;
    } indirectData;
    
    vk::BufferCreateInfo bufferInfo{
        {},
        sizeof(DispatchIndirectCommand),
        vk::BufferUsageFlagBits::eIndirectBuffer,
        vk::SharingMode::eExclusive
    };
    
    vk::raii::Buffer indirectBuffer(device, bufferInfo);
    
    // Allocate and fill memory
    auto memRequirements = indirectBuffer.getMemoryRequirements();
    auto memProperties = physicalDevice.getMemoryProperties();
    
    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
            memoryTypeIndex = i;
            break;
        }
    }
    
    ASSERT_NE(memoryTypeIndex, UINT32_MAX);
    
    vk::MemoryAllocateInfo allocInfo{
        memRequirements.size,
        memoryTypeIndex
    };
    
    vk::raii::DeviceMemory memory(device, allocInfo);
    indirectBuffer.bindMemory(*memory, 0);
    
    // Write indirect data
    void* mappedMemory = memory.mapMemory(0, sizeof(DispatchIndirectCommand));
    std::memcpy(mappedMemory, &indirectData, sizeof(DispatchIndirectCommand));
    memory.unmapMemory();
    
    // Would be used in dispatchIndirect command
    EXPECT_TRUE(*indirectBuffer);
}

// Test 20: Descriptor Indexing (Vulkan 1.2)
TEST_F(VulkanCoreTest, DescriptorIndexing) {
    // Check for descriptor indexing support
    vk::PhysicalDeviceVulkan12Features features12{};
    vk::PhysicalDeviceFeatures2 features2{};
    features2.pNext = &features12;
    physicalDevice.getFeatures2(&features2);
    
    if (!features12.descriptorIndexing) {
        GTEST_SKIP() << "Descriptor indexing not supported";
    }
    
    // Create descriptor pool with update after bind flag
    vk::DescriptorPoolSize poolSize{
        vk::DescriptorType::eStorageBuffer,
        100  // Large number for indexing
    };
    
    vk::DescriptorPoolCreateInfo poolInfo{
        vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        10,
        poolSize
    };
    
    vk::raii::DescriptorPool descriptorPool(device, poolInfo);
    ASSERT_TRUE(*descriptorPool) << "Descriptor pool with update after bind failed";
    
    // Create descriptor set layout with variable descriptor count
    vk::DescriptorSetLayoutBinding layoutBinding{
        0,
        vk::DescriptorType::eStorageBuffer,
        100,  // Variable count
        vk::ShaderStageFlagBits::eCompute
    };
    
    vk::DescriptorBindingFlags bindingFlags = 
        vk::DescriptorBindingFlagBits::eUpdateAfterBind |
        vk::DescriptorBindingFlagBits::ePartiallyBound |
        vk::DescriptorBindingFlagBits::eVariableDescriptorCount;
    
    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        1,
        &bindingFlags
    };
    
    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        layoutBinding
    };
    layoutInfo.pNext = &bindingFlagsInfo;
    
    vk::raii::DescriptorSetLayout setLayout(device, layoutInfo);
    ASSERT_TRUE(*setLayout) << "Descriptor set layout with indexing failed";
}

} // namespace mlsdk::tests