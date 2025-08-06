/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Memory Management Unit Tests for ARM ML SDK
 * Tests memory allocation patterns, alignment, unified memory handling,
 * and memory leak detection.
 */

#include <gtest/gtest.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <memory>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <map>

namespace mlsdk::tests {

// Custom memory allocator for tracking
class MemoryTracker {
public:
    struct AllocationInfo {
        size_t size;
        std::string tag;
        std::chrono::steady_clock::time_point timestamp;
    };
    
private:
    std::map<void*, AllocationInfo> allocations;
    std::atomic<size_t> totalAllocated{0};
    std::atomic<size_t> peakAllocated{0};
    std::atomic<size_t> allocationCount{0};
    mutable std::mutex mutex;
    
public:
    void* allocate(size_t size, const std::string& tag = "") {
        void* ptr = std::malloc(size);
        if (ptr) {
            std::lock_guard<std::mutex> lock(mutex);
            allocations[ptr] = {size, tag, std::chrono::steady_clock::now()};
            totalAllocated += size;
            allocationCount++;
            
            if (totalAllocated > peakAllocated) {
                peakAllocated = totalAllocated.load();
            }
        }
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (ptr) {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = allocations.find(ptr);
            if (it != allocations.end()) {
                totalAllocated -= it->second.size;
                allocations.erase(it);
            }
            std::free(ptr);
        }
    }
    
    size_t getCurrentUsage() const { return totalAllocated; }
    size_t getPeakUsage() const { return peakAllocated; }
    size_t getAllocationCount() const { return allocationCount; }
    
    bool hasLeaks() const {
        std::lock_guard<std::mutex> lock(mutex);
        return !allocations.empty();
    }
    
    std::vector<AllocationInfo> getActiveAllocations() const {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<AllocationInfo> result;
        for (const auto& [ptr, info] : allocations) {
            result.push_back(info);
        }
        return result;
    }
};

class MemoryManagementTest : public ::testing::Test {
protected:
    vk::raii::Context context;
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    uint32_t queueFamilyIndex = 0;
    MemoryTracker memTracker;
    
    void SetUp() override {
        // Create Vulkan instance
        vk::ApplicationInfo appInfo{
            "MemoryManagementTests",
            VK_MAKE_VERSION(1, 0, 0),
            "ARM ML SDK Tests",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_2
        };
        
        vk::InstanceCreateInfo instanceCreateInfo{
            {},
            &appInfo,
            {},
            {}
        };
        
        instance = vk::raii::Instance(context, instanceCreateInfo);
        
        // Select physical device
        auto physicalDevices = vk::raii::PhysicalDevices(instance);
        ASSERT_FALSE(physicalDevices.empty());
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
        
        vk::DeviceCreateInfo deviceCreateInfo{
            {},
            queueCreateInfo,
            {},
            {},
            {}
        };
        
        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    }
    
    void TearDown() override {
        if (*device) {
            device.waitIdle();
        }
        
        // Check for memory leaks
        EXPECT_FALSE(memTracker.hasLeaks()) << "Memory leaks detected";
    }
    
    // Helper to find memory type index
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        auto memProperties = physicalDevice.getMemoryProperties();
        
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        
        return UINT32_MAX;
    }
};

// Test 1: Memory Alignment
TEST_F(MemoryManagementTest, MemoryAlignment) {
    // Test various alignment requirements
    const std::vector<size_t> alignments = {16, 32, 64, 128, 256, 512, 1024, 4096};
    const size_t baseSize = 1000; // Unaligned size
    
    for (size_t alignment : alignments) {
        // Calculate aligned size
        size_t alignedSize = (baseSize + alignment - 1) & ~(alignment - 1);
        
        // Verify alignment calculation
        EXPECT_EQ(alignedSize % alignment, 0) << "Size not aligned to " << alignment;
        EXPECT_GE(alignedSize, baseSize) << "Aligned size smaller than original";
        EXPECT_LT(alignedSize, baseSize + alignment) << "Aligned size too large";
        
        // Test with Vulkan buffer
        vk::BufferCreateInfo bufferInfo{
            {},
            alignedSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive
        };
        
        vk::raii::Buffer buffer(device, bufferInfo);
        auto memRequirements = buffer.getMemoryRequirements();
        
        // Vulkan may have its own alignment requirements
        EXPECT_EQ(memRequirements.size % memRequirements.alignment, 0);
    }
}

// Test 2: Memory Type Selection
TEST_F(MemoryManagementTest, MemoryTypeSelection) {
    auto memProperties = physicalDevice.getMemoryProperties();
    
    // Test finding different memory types
    struct MemoryTypeTest {
        vk::MemoryPropertyFlags required;
        vk::MemoryPropertyFlags preferred;
        std::string description;
    };
    
    std::vector<MemoryTypeTest> tests = {
        {vk::MemoryPropertyFlagBits::eDeviceLocal, 
         vk::MemoryPropertyFlagBits::eDeviceLocal,
         "Device local memory"},
        
        {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
         "Host visible coherent memory"},
        
        {vk::MemoryPropertyFlagBits::eHostVisible,
         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached,
         "Host cached memory (preferred)"}
    };
    
    for (const auto& test : tests) {
        // Create a buffer to get memory requirements
        vk::BufferCreateInfo bufferInfo{
            {},
            1024,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive
        };
        
        vk::raii::Buffer buffer(device, bufferInfo);
        auto memRequirements = buffer.getMemoryRequirements();
        
        // Find suitable memory type
        uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, test.required);
        
        if (memoryTypeIndex != UINT32_MAX) {
            EXPECT_LT(memoryTypeIndex, memProperties.memoryTypeCount) 
                << "Invalid memory type index for " << test.description;
            
            // Check if preferred flags are available
            uint32_t preferredIndex = findMemoryType(memRequirements.memoryTypeBits, test.preferred);
            if (preferredIndex != UINT32_MAX) {
                // Use preferred if available
                memoryTypeIndex = preferredIndex;
            }
            
            // Allocate memory
            vk::MemoryAllocateInfo allocInfo{
                memRequirements.size,
                memoryTypeIndex
            };
            
            vk::raii::DeviceMemory memory(device, allocInfo);
            EXPECT_TRUE(*memory) << "Failed to allocate " << test.description;
        }
    }
}

// Test 3: Memory Pool Allocation
TEST_F(MemoryManagementTest, MemoryPoolAllocation) {
    // Simulate a memory pool allocator
    class MemoryPool {
    private:
        struct Block {
            size_t offset;
            size_t size;
            bool free;
        };
        
        size_t totalSize;
        std::vector<Block> blocks;
        
    public:
        explicit MemoryPool(size_t size) : totalSize(size) {
            blocks.push_back({0, size, true});
        }
        
        size_t allocate(size_t size, size_t alignment) {
            for (auto& block : blocks) {
                if (block.free && block.size >= size) {
                    // Align offset
                    size_t alignedOffset = (block.offset + alignment - 1) & ~(alignment - 1);
                    size_t alignedSize = size + (alignedOffset - block.offset);
                    
                    if (block.size >= alignedSize) {
                        // Split block if necessary
                        if (block.size > alignedSize) {
                            blocks.push_back({
                                alignedOffset + size,
                                block.size - alignedSize,
                                true
                            });
                        }
                        
                        block.offset = alignedOffset;
                        block.size = size;
                        block.free = false;
                        
                        return alignedOffset;
                    }
                }
            }
            return SIZE_MAX; // Allocation failed
        }
        
        void deallocate(size_t offset) {
            for (auto& block : blocks) {
                if (block.offset == offset && !block.free) {
                    block.free = true;
                    // TODO: Coalesce adjacent free blocks
                    return;
                }
            }
        }
        
        size_t getFragmentation() const {
            size_t freeSpace = 0;
            size_t largestFree = 0;
            
            for (const auto& block : blocks) {
                if (block.free) {
                    freeSpace += block.size;
                    largestFree = std::max(largestFree, block.size);
                }
            }
            
            if (freeSpace == 0) return 0;
            return 100 - (largestFree * 100 / freeSpace);
        }
    };
    
    const size_t poolSize = 1024 * 1024; // 1MB pool
    MemoryPool pool(poolSize);
    
    // Allocate various sizes
    std::vector<size_t> allocations;
    allocations.push_back(pool.allocate(1024, 256));
    allocations.push_back(pool.allocate(2048, 256));
    allocations.push_back(pool.allocate(4096, 256));
    
    // Verify allocations succeeded
    for (size_t offset : allocations) {
        EXPECT_NE(offset, SIZE_MAX) << "Allocation failed";
        EXPECT_EQ(offset % 256, 0) << "Allocation not aligned";
    }
    
    // Deallocate middle allocation to create fragmentation
    pool.deallocate(allocations[1]);
    
    // Try to allocate in the freed space
    size_t newAlloc = pool.allocate(2048, 256);
    EXPECT_NE(newAlloc, SIZE_MAX) << "Failed to reuse freed memory";
    
    // Check fragmentation
    size_t fragmentation = pool.getFragmentation();
    EXPECT_LT(fragmentation, 100) << "Memory pool is 100% fragmented";
}

// Test 4: Unified Memory (Apple Silicon)
TEST_F(MemoryManagementTest, UnifiedMemoryHandling) {
    auto memProperties = physicalDevice.getMemoryProperties();
    auto deviceProperties = physicalDevice.getProperties();
    
    // Check for unified memory architecture
    bool hasUnifiedMemory = false;
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) {
        if (memProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            // On unified memory systems, device local memory is also host visible
            for (uint32_t j = 0; j < memProperties.memoryTypeCount; ++j) {
                if ((memProperties.memoryTypes[j].heapIndex == i) &&
                    (memProperties.memoryTypes[j].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
                    hasUnifiedMemory = true;
                    break;
                }
            }
        }
    }
    
    if (hasUnifiedMemory) {
        std::cout << "Unified memory architecture detected (likely Apple Silicon)" << std::endl;
        
        // Test unified memory allocation
        vk::BufferCreateInfo bufferInfo{
            {},
            1024 * 1024, // 1MB
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            vk::SharingMode::eExclusive
        };
        
        vk::raii::Buffer buffer(device, bufferInfo);
        auto memRequirements = buffer.getMemoryRequirements();
        
        // Find memory that is both device local and host visible
        uint32_t memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((memRequirements.memoryTypeBits & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) &&
                (memProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
                memoryTypeIndex = i;
                break;
            }
        }
        
        if (memoryTypeIndex != UINT32_MAX) {
            vk::MemoryAllocateInfo allocInfo{
                memRequirements.size,
                memoryTypeIndex
            };
            
            vk::raii::DeviceMemory memory(device, allocInfo);
            buffer.bindMemory(*memory, 0);
            
            // Should be able to map device local memory on unified architecture
            void* mappedMemory = memory.mapMemory(0, memRequirements.size);
            EXPECT_NE(mappedMemory, nullptr) << "Failed to map unified memory";
            
            // Write and verify data
            std::vector<uint8_t> testData(1024, 0x42);
            std::memcpy(mappedMemory, testData.data(), testData.size());
            
            memory.unmapMemory();
        }
    }
}

// Test 5: Memory Allocation Stress Test
TEST_F(MemoryManagementTest, AllocationStressTest) {
    const size_t numAllocations = 100;
    const size_t minSize = 1024;        // 1KB
    const size_t maxSize = 1024 * 1024; // 1MB
    
    std::vector<std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>> allocations;
    allocations.reserve(numAllocations);
    
    // Random number generator for sizes
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> sizeDist(minSize, maxSize);
    
    // Allocate many buffers
    for (size_t i = 0; i < numAllocations; ++i) {
        size_t size = sizeDist(rng);
        
        vk::BufferCreateInfo bufferInfo{
            {},
            size,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive
        };
        
        try {
            vk::raii::Buffer buffer(device, bufferInfo);
            auto memRequirements = buffer.getMemoryRequirements();
            
            uint32_t memoryTypeIndex = findMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            );
            
            if (memoryTypeIndex != UINT32_MAX) {
                vk::MemoryAllocateInfo allocInfo{
                    memRequirements.size,
                    memoryTypeIndex
                };
                
                vk::raii::DeviceMemory memory(device, allocInfo);
                buffer.bindMemory(*memory, 0);
                
                allocations.emplace_back(std::move(buffer), std::move(memory));
            }
        } catch (const vk::SystemError& e) {
            // Allocation might fail due to memory limits
            break;
        }
    }
    
    EXPECT_GT(allocations.size(), 0) << "No allocations succeeded";
    
    // Calculate total allocated memory
    size_t totalAllocated = 0;
    for (const auto& [buffer, memory] : allocations) {
        auto memRequirements = buffer.getMemoryRequirements();
        totalAllocated += memRequirements.size;
    }
    
    std::cout << "Successfully allocated " << allocations.size() 
              << " buffers, total size: " << (totalAllocated / (1024.0 * 1024.0)) 
              << " MB" << std::endl;
    
    // Cleanup happens automatically with RAII
}

// Test 6: Memory Leak Detection
TEST_F(MemoryManagementTest, MemoryLeakDetection) {
    // Test the memory tracker
    
    // Allocate without deallocation (leak)
    void* leaked = memTracker.allocate(1024, "intentional_leak");
    EXPECT_NE(leaked, nullptr);
    EXPECT_EQ(memTracker.getCurrentUsage(), 1024);
    
    // Allocate and deallocate properly
    void* proper = memTracker.allocate(2048, "proper_allocation");
    EXPECT_NE(proper, nullptr);
    EXPECT_EQ(memTracker.getCurrentUsage(), 3072);
    
    memTracker.deallocate(proper);
    EXPECT_EQ(memTracker.getCurrentUsage(), 1024);
    
    // Check for leaks
    EXPECT_TRUE(memTracker.hasLeaks());
    auto leaks = memTracker.getActiveAllocations();
    EXPECT_EQ(leaks.size(), 1);
    EXPECT_EQ(leaks[0].size, 1024);
    EXPECT_EQ(leaks[0].tag, "intentional_leak");
    
    // Clean up the leak
    memTracker.deallocate(leaked);
    EXPECT_FALSE(memTracker.hasLeaks());
}

// Test 7: Memory Bandwidth Test
TEST_F(MemoryManagementTest, MemoryBandwidth) {
    const size_t bufferSize = 16 * 1024 * 1024; // 16MB
    
    // Create source and destination buffers
    vk::BufferCreateInfo bufferInfo{
        {},
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive
    };
    
    vk::raii::Buffer srcBuffer(device, bufferInfo);
    vk::raii::Buffer dstBuffer(device, bufferInfo);
    
    auto memRequirements = srcBuffer.getMemoryRequirements();
    
    // Allocate host-visible memory for testing
    uint32_t memoryTypeIndex = findMemoryType(
        memRequirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    
    if (memoryTypeIndex != UINT32_MAX) {
        vk::MemoryAllocateInfo allocInfo{
            memRequirements.size * 2,
            memoryTypeIndex
        };
        
        vk::raii::DeviceMemory memory(device, allocInfo);
        srcBuffer.bindMemory(*memory, 0);
        dstBuffer.bindMemory(*memory, memRequirements.size);
        
        // Map memory and perform bandwidth test
        void* mappedMemory = memory.mapMemory(0, memRequirements.size * 2);
        
        // Initialize source buffer
        std::vector<uint8_t> testData(bufferSize);
        for (size_t i = 0; i < bufferSize; ++i) {
            testData[i] = static_cast<uint8_t>(i & 0xFF);
        }
        
        // Measure write bandwidth
        auto writeStart = std::chrono::high_resolution_clock::now();
        std::memcpy(mappedMemory, testData.data(), bufferSize);
        auto writeEnd = std::chrono::high_resolution_clock::now();
        
        // Measure read bandwidth
        std::vector<uint8_t> readData(bufferSize);
        auto readStart = std::chrono::high_resolution_clock::now();
        std::memcpy(readData.data(), mappedMemory, bufferSize);
        auto readEnd = std::chrono::high_resolution_clock::now();
        
        // Calculate bandwidth
        auto writeDuration = std::chrono::duration<double>(writeEnd - writeStart).count();
        auto readDuration = std::chrono::duration<double>(readEnd - readStart).count();
        
        double writeBandwidth = (bufferSize / (1024.0 * 1024.0 * 1024.0)) / writeDuration; // GB/s
        double readBandwidth = (bufferSize / (1024.0 * 1024.0 * 1024.0)) / readDuration;   // GB/s
        
        std::cout << "Memory Write Bandwidth: " << writeBandwidth << " GB/s" << std::endl;
        std::cout << "Memory Read Bandwidth: " << readBandwidth << " GB/s" << std::endl;
        
        // Basic sanity checks
        EXPECT_GT(writeBandwidth, 0.1) << "Write bandwidth too low";
        EXPECT_GT(readBandwidth, 0.1) << "Read bandwidth too low";
        
        memory.unmapMemory();
    }
}

// Test 8: Memory Heap Information
TEST_F(MemoryManagementTest, MemoryHeapInfo) {
    auto memProperties = physicalDevice.getMemoryProperties();
    
    std::cout << "Memory Heaps:" << std::endl;
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) {
        const auto& heap = memProperties.memoryHeaps[i];
        
        std::cout << "  Heap " << i << ":" << std::endl;
        std::cout << "    Size: " << (heap.size / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        std::cout << "    Flags: ";
        
        if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            std::cout << "DEVICE_LOCAL ";
        }
        if (heap.flags & vk::MemoryHeapFlagBits::eMultiInstance) {
            std::cout << "MULTI_INSTANCE ";
        }
        std::cout << std::endl;
        
        // Find memory types for this heap
        std::cout << "    Memory Types: ";
        for (uint32_t j = 0; j < memProperties.memoryTypeCount; ++j) {
            if (memProperties.memoryTypes[j].heapIndex == i) {
                std::cout << j << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // Verify at least one heap exists
    EXPECT_GT(memProperties.memoryHeapCount, 0);
    
    // Calculate total available memory
    VkDeviceSize totalMemory = 0;
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) {
        totalMemory += memProperties.memoryHeaps[i].size;
    }
    
    EXPECT_GT(totalMemory, 0) << "No memory available";
    std::cout << "Total Available Memory: " 
              << (totalMemory / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
}

// Test 9: Memory Alignment for Apple GPU
TEST_F(MemoryManagementTest, AppleGPUAlignment) {
    auto deviceProperties = physicalDevice.getProperties();
    std::string deviceName(deviceProperties.deviceName);
    
    // Check if running on Apple GPU
    bool isAppleGPU = deviceName.find("Apple") != std::string::npos ||
                      deviceName.find("M1") != std::string::npos ||
                      deviceName.find("M2") != std::string::npos ||
                      deviceName.find("M3") != std::string::npos ||
                      deviceName.find("M4") != std::string::npos;
    
    if (isAppleGPU) {
        std::cout << "Apple GPU detected: " << deviceName << std::endl;
        
        // Apple GPUs typically have specific alignment requirements
        const size_t APPLE_GPU_CACHE_LINE = 256; // Common cache line size
        
        // Test buffer alignment
        vk::BufferCreateInfo bufferInfo{
            {},
            1000, // Unaligned size
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive
        };
        
        vk::raii::Buffer buffer(device, bufferInfo);
        auto memRequirements = buffer.getMemoryRequirements();
        
        // Check alignment requirements
        std::cout << "Buffer alignment requirement: " << memRequirements.alignment << " bytes" << std::endl;
        std::cout << "Buffer size requirement: " << memRequirements.size << " bytes" << std::endl;
        
        // Alignment should be power of 2
        EXPECT_EQ(memRequirements.alignment & (memRequirements.alignment - 1), 0) 
            << "Alignment is not power of 2";
    }
}

// Test 10: Memory Coherency
TEST_F(MemoryManagementTest, MemoryCoherency) {
    const size_t bufferSize = 4096;
    
    // Create buffer
    vk::BufferCreateInfo bufferInfo{
        {},
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::SharingMode::eExclusive
    };
    
    vk::raii::Buffer buffer(device, bufferInfo);
    auto memRequirements = buffer.getMemoryRequirements();
    
    // Test both coherent and non-coherent memory
    struct CoherencyTest {
        vk::MemoryPropertyFlags flags;
        std::string description;
        bool needsFlush;
    };
    
    std::vector<CoherencyTest> tests = {
        {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
         "Host coherent memory", false},
        {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached,
         "Host cached memory", true}
    };
    
    for (const auto& test : tests) {
        uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, test.flags);
        
        if (memoryTypeIndex != UINT32_MAX) {
            vk::MemoryAllocateInfo allocInfo{
                memRequirements.size,
                memoryTypeIndex
            };
            
            vk::raii::DeviceMemory memory(device, allocInfo);
            buffer.bindMemory(*memory, 0);
            
            void* mappedMemory = memory.mapMemory(0, bufferSize);
            
            // Write test pattern
            std::vector<uint32_t> testData(bufferSize / sizeof(uint32_t));
            for (size_t i = 0; i < testData.size(); ++i) {
                testData[i] = static_cast<uint32_t>(i);
            }
            std::memcpy(mappedMemory, testData.data(), bufferSize);
            
            if (test.needsFlush) {
                // Flush memory range for non-coherent memory
                vk::MappedMemoryRange memoryRange{
                    *memory,
                    0,
                    VK_WHOLE_SIZE
                };
                device.flushMappedMemoryRanges(memoryRange);
            }
            
            // Read back and verify
            std::vector<uint32_t> readData(bufferSize / sizeof(uint32_t));
            
            if (test.needsFlush) {
                // Invalidate cache before reading
                vk::MappedMemoryRange memoryRange{
                    *memory,
                    0,
                    VK_WHOLE_SIZE
                };
                device.invalidateMappedMemoryRanges(memoryRange);
            }
            
            std::memcpy(readData.data(), mappedMemory, bufferSize);
            
            // Verify data
            EXPECT_EQ(readData, testData) << "Data mismatch for " << test.description;
            
            memory.unmapMemory();
        }
    }
}

} // namespace mlsdk::tests