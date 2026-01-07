/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compute Pipeline Implementation with Subgroup Operation Support
 * Provides advanced Vulkan 1.4 compute features with automatic fallback paths
 */

#include "compute.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace mlsdk {

ComputePipelineManager::ComputePipelineManager(
    vk::raii::Device& device,
    vk::raii::PhysicalDevice& physicalDevice,
    uint32_t computeQueueFamilyIndex
) : _device(device)
  , _physicalDevice(physicalDevice)
  , _computeQueueFamilyIndex(computeQueueFamilyIndex)
  , _pipelineCache(nullptr)
{
    // Query subgroup capabilities on construction
    querySubgroupCapabilities();

    // Initialize pipeline cache
    initializePipelineCache();
}

ComputePipelineManager::~ComputePipelineManager() {
    clearPipelines();
}

ComputePipelineManager::ComputePipelineManager(ComputePipelineManager&& other) noexcept
    : _device(other._device)
    , _physicalDevice(other._physicalDevice)
    , _computeQueueFamilyIndex(other._computeQueueFamilyIndex)
    , _subgroupCaps(std::move(other._subgroupCaps))
    , _pipelineCache(std::move(other._pipelineCache))
    , _pipelines(std::move(other._pipelines))
{
}

ComputePipelineManager& ComputePipelineManager::operator=(ComputePipelineManager&& other) noexcept {
    if (this != &other) {
        _computeQueueFamilyIndex = other._computeQueueFamilyIndex;
        _subgroupCaps = std::move(other._subgroupCaps);
        _pipelineCache = std::move(other._pipelineCache);
        _pipelines = std::move(other._pipelines);
    }
    return *this;
}

void ComputePipelineManager::querySubgroupCapabilities() {
    // Query subgroup properties using Vulkan 1.1+ API
    vk::PhysicalDeviceSubgroupProperties subgroupProperties;
    vk::PhysicalDeviceProperties2 properties2;
    properties2.pNext = &subgroupProperties;

    _physicalDevice.getProperties2(&properties2);

    // Store subgroup properties
    _subgroupCaps.subgroupSize = subgroupProperties.subgroupSize;
    _subgroupCaps.supportedStages = static_cast<VkShaderStageFlags>(subgroupProperties.supportedStages);
    _subgroupCaps.supportedOperations = static_cast<VkSubgroupFeatureFlags>(subgroupProperties.supportedOperations);
    _subgroupCaps.quadOperationsInAllStages = subgroupProperties.quadOperationsInAllStages;

    // Parse individual operation support
    _subgroupCaps.hasBasic = _subgroupCaps.supportsOperation(VK_SUBGROUP_FEATURE_BASIC_BIT);
    _subgroupCaps.hasVote = _subgroupCaps.supportsOperation(VK_SUBGROUP_FEATURE_VOTE_BIT);
    _subgroupCaps.hasArithmetic = _subgroupCaps.supportsOperation(VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    _subgroupCaps.hasBallot = _subgroupCaps.supportsOperation(VK_SUBGROUP_FEATURE_BALLOT_BIT);
    _subgroupCaps.hasShuffle = _subgroupCaps.supportsOperation(VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
    _subgroupCaps.hasShuffleRelative = _subgroupCaps.supportsOperation(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT);
    _subgroupCaps.hasClustered = _subgroupCaps.supportsOperation(VK_SUBGROUP_FEATURE_CLUSTERED_BIT);
    _subgroupCaps.hasQuad = _subgroupCaps.supportsOperation(VK_SUBGROUP_FEATURE_QUAD_BIT);

    // Check for NV partitioned subgroups if extension is available
    _subgroupCaps.hasPartitionedNV = _subgroupCaps.supportsOperation(
        static_cast<VkSubgroupFeatureFlagBits>(0x00000100)  // VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV
    );

    // Determine if fallback is required
    // Minimum requirement: basic subgroup operations in compute stage
    _subgroupCaps.requiresFallback = !detectMinimumSubgroupSupport();

    // Log subgroup capabilities for debugging
    std::cout << "[ComputePipeline] Subgroup capabilities detected:" << std::endl;
    std::cout << "  Subgroup size: " << _subgroupCaps.subgroupSize << std::endl;
    std::cout << "  Compute stage support: " << (_subgroupCaps.supportsComputeStage() ? "YES" : "NO") << std::endl;
    std::cout << "  Basic operations: " << (_subgroupCaps.hasBasic ? "YES" : "NO") << std::endl;
    std::cout << "  Vote operations: " << (_subgroupCaps.hasVote ? "YES" : "NO") << std::endl;
    std::cout << "  Arithmetic operations: " << (_subgroupCaps.hasArithmetic ? "YES" : "NO") << std::endl;
    std::cout << "  Ballot operations: " << (_subgroupCaps.hasBallot ? "YES" : "NO") << std::endl;
    std::cout << "  Shuffle operations: " << (_subgroupCaps.hasShuffle ? "YES" : "NO") << std::endl;
    std::cout << "  Clustered operations: " << (_subgroupCaps.hasClustered ? "YES" : "NO") << std::endl;
    std::cout << "  Quad operations: " << (_subgroupCaps.hasQuad ? "YES" : "NO") << std::endl;
    std::cout << "  Requires fallback path: " << (_subgroupCaps.requiresFallback ? "YES" : "NO") << std::endl;
}

bool ComputePipelineManager::detectMinimumSubgroupSupport() const {
    // Minimum requirements for subgroup optimization:
    // 1. Non-zero subgroup size
    // 2. Compute stage support
    // 3. At least basic operations supported
    return _subgroupCaps.subgroupSize > 0 &&
           _subgroupCaps.supportsComputeStage() &&
           _subgroupCaps.hasBasic;
}

void ComputePipelineManager::initializePipelineCache() {
    vk::PipelineCacheCreateInfo cacheInfo{};

    // RAII pattern: use unique_ptr for automatic cleanup
    _pipelineCache = std::make_unique<vk::raii::PipelineCache>(_device, cacheInfo);
}

bool ComputePipelineManager::createPipeline(
    const ComputePipelineConfig& config,
    const std::string& pipelineName
) {
    // Load shader SPIR-V
    auto spirvCode = loadShaderSPIRV(config.shaderPath);
    if (spirvCode.empty()) {
        std::cerr << "[ComputePipeline] Failed to load shader: " << config.shaderPath << std::endl;
        return false;
    }

    // Create shader module
    auto shaderModule = createShaderModule(spirvCode);

    // Determine if this shader uses subgroup operations
    bool usesSubgroups = config.useSubgroupOperations &&
                         !_subgroupCaps.requiresFallback &&
                         SubgroupUtils::shaderUsesSubgroupOps(spirvCode);

    // Create descriptor set layout
    vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo{};
    if (!config.descriptorBindings.empty()) {
        descriptorLayoutInfo.setBindings(config.descriptorBindings);
    }

    auto descriptorSetLayout = std::make_unique<vk::raii::DescriptorSetLayout>(
        _device, descriptorLayoutInfo
    );

    // Create pipeline layout
    std::vector<vk::DescriptorSetLayout> setLayouts = {**descriptorSetLayout};

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.setSetLayouts(setLayouts);
    if (!config.pushConstantRanges.empty()) {
        layoutInfo.setPushConstantRanges(config.pushConstantRanges);
    }

    auto pipelineLayout = std::make_unique<vk::raii::PipelineLayout>(_device, layoutInfo);

    // Create specialization constants for subgroup parameters
    SubgroupSpecializationData specData = createSubgroupSpecData(config.workgroupSizeX);
    auto specEntries = getSubgroupSpecializationEntries();

    vk::SpecializationInfo specInfo{
        static_cast<uint32_t>(specEntries.size()),
        specEntries.data(),
        sizeof(SubgroupSpecializationData),
        &specData
    };

    // Create compute pipeline
    vk::PipelineShaderStageCreateInfo stageInfo{
        {},
        vk::ShaderStageFlagBits::eCompute,
        *shaderModule,
        config.entryPoint.c_str(),
        &specInfo
    };

    vk::ComputePipelineCreateInfo pipelineInfo{
        {},
        stageInfo,
        **pipelineLayout
    };

    try {
        // Create pipeline with cache
        auto pipelines = vk::raii::Pipelines(_device, **_pipelineCache, pipelineInfo);

        if (pipelines.empty()) {
            std::cerr << "[ComputePipeline] Pipeline creation returned empty" << std::endl;
            return false;
        }

        // Store in pipelines map
        PipelineEntry entry;
        entry.pipeline = std::make_unique<vk::raii::Pipeline>(std::move(pipelines[0]));
        entry.layout = std::move(pipelineLayout);
        entry.descriptorSetLayout = std::move(descriptorSetLayout);
        entry.usesSubgroups = usesSubgroups;

        _pipelines[pipelineName] = std::move(entry);

        std::cout << "[ComputePipeline] Created pipeline '" << pipelineName << "'"
                  << (usesSubgroups ? " (with subgroup operations)" : " (standard)") << std::endl;

        return true;

    } catch (const vk::SystemError& e) {
        std::cerr << "[ComputePipeline] Failed to create pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool ComputePipelineManager::createPipelineWithFallback(
    const std::string& subgroupShaderPath,
    const std::string& fallbackShaderPath,
    const ComputePipelineConfig& baseConfig,
    const std::string& pipelineName
) {
    // Check if subgroup operations are supported
    if (!_subgroupCaps.requiresFallback) {
        // Try to create pipeline with subgroup shader
        ComputePipelineConfig config = baseConfig;
        config.shaderPath = subgroupShaderPath;
        config.useSubgroupOperations = true;

        std::cout << "[ComputePipeline] Attempting subgroup-optimized pipeline for '"
                  << pipelineName << "'" << std::endl;

        if (createPipeline(config, pipelineName)) {
            return true;
        }

        std::cout << "[ComputePipeline] Subgroup pipeline failed, falling back to standard" << std::endl;
    } else {
        std::cout << "[ComputePipeline] Subgroup operations not supported, using fallback for '"
                  << pipelineName << "'" << std::endl;
    }

    // Use fallback shader
    ComputePipelineConfig fallbackConfig = baseConfig;
    fallbackConfig.shaderPath = fallbackShaderPath;
    fallbackConfig.useSubgroupOperations = false;

    return createPipeline(fallbackConfig, pipelineName);
}

vk::Pipeline ComputePipelineManager::getPipeline(const std::string& name) const {
    auto it = _pipelines.find(name);
    if (it != _pipelines.end() && it->second.pipeline) {
        return **(it->second.pipeline);
    }
    return VK_NULL_HANDLE;
}

vk::PipelineLayout ComputePipelineManager::getPipelineLayout(const std::string& name) const {
    auto it = _pipelines.find(name);
    if (it != _pipelines.end() && it->second.layout) {
        return **(it->second.layout);
    }
    return VK_NULL_HANDLE;
}

void ComputePipelineManager::recordDispatch(
    vk::raii::CommandBuffer& cmdBuffer,
    const std::string& pipelineName,
    const DispatchConfig& dispatch,
    const void* pushConstantData,
    uint32_t pushConstantSize
) {
    auto it = _pipelines.find(pipelineName);
    if (it == _pipelines.end() || !it->second.pipeline) {
        std::cerr << "[ComputePipeline] Pipeline not found: " << pipelineName << std::endl;
        return;
    }

    const auto& entry = it->second;

    // Bind pipeline
    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, **(entry.pipeline));

    // Push constants if provided
    if (pushConstantData && pushConstantSize > 0) {
        cmdBuffer.pushConstants(
            **(entry.layout),
            vk::ShaderStageFlagBits::eCompute,
            0,
            pushConstantSize,
            pushConstantData
        );
    }

    // Dispatch
    if (dispatch.indirect) {
        cmdBuffer.dispatchIndirect(dispatch.indirectBuffer, dispatch.indirectOffset);
    } else {
        cmdBuffer.dispatch(dispatch.groupCountX, dispatch.groupCountY, dispatch.groupCountZ);
    }
}

void ComputePipelineManager::recordSubgroupDispatch(
    vk::raii::CommandBuffer& cmdBuffer,
    const std::string& pipelineName,
    const DispatchConfig& dispatch,
    bool requiresSubgroupBarrier
) {
    auto it = _pipelines.find(pipelineName);
    if (it == _pipelines.end() || !it->second.pipeline) {
        std::cerr << "[ComputePipeline] Pipeline not found: " << pipelineName << std::endl;
        return;
    }

    const auto& entry = it->second;

    // Bind pipeline
    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, **(entry.pipeline));

    // Add memory barrier before dispatch if using subgroups
    if (entry.usesSubgroups && requiresSubgroupBarrier) {
        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead
        };

        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {},
            barrier,
            {},
            {}
        );
    }

    // Dispatch
    if (dispatch.indirect) {
        cmdBuffer.dispatchIndirect(dispatch.indirectBuffer, dispatch.indirectOffset);
    } else {
        // Align dispatch count to subgroup size for optimal performance
        uint32_t alignedX = dispatch.groupCountX;
        if (entry.usesSubgroups && _subgroupCaps.subgroupSize > 0) {
            // Ensure full subgroups are utilized
            alignedX = SubgroupUtils::alignToSubgroup(
                dispatch.groupCountX,
                _subgroupCaps.subgroupSize / 4  // Account for typical workgroup granularity
            );
        }

        cmdBuffer.dispatch(alignedX, dispatch.groupCountY, dispatch.groupCountZ);
    }

    // Add memory barrier after dispatch for subgroup operations
    if (entry.usesSubgroups) {
        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead
        };

        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            {},
            barrier,
            {},
            {}
        );
    }
}

std::vector<vk::SpecializationMapEntry> ComputePipelineManager::getSubgroupSpecializationEntries() const {
    return {
        // Subgroup size constant ID 0
        {0, offsetof(SubgroupSpecializationData, subgroupSize), sizeof(uint32_t)},
        // Use subgroup ops flag constant ID 1
        {1, offsetof(SubgroupSpecializationData, useSubgroupOps), sizeof(uint32_t)},
        // Workgroup size constant ID 2
        {2, offsetof(SubgroupSpecializationData, workgroupSize), sizeof(uint32_t)}
    };
}

ComputePipelineManager::SubgroupSpecializationData
ComputePipelineManager::createSubgroupSpecData(uint32_t workgroupSize) const {
    SubgroupSpecializationData data;
    data.subgroupSize = _subgroupCaps.subgroupSize > 0 ? _subgroupCaps.subgroupSize : 32;
    data.useSubgroupOps = _subgroupCaps.requiresFallback ? 0 : 1;
    data.workgroupSize = _subgroupCaps.getOptimalWorkgroupSize(workgroupSize);
    return data;
}

void ComputePipelineManager::clearPipelines() {
    // Clear all pipeline entries - unique_ptrs handle cleanup
    _pipelines.clear();
}

vk::PipelineCache ComputePipelineManager::getPipelineCache() const {
    return _pipelineCache ? **_pipelineCache : VK_NULL_HANDLE;
}

std::vector<uint32_t> ComputePipelineManager::loadShaderSPIRV(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "[ComputePipeline] Failed to open shader file: " << path << std::endl;
        return {};
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize % sizeof(uint32_t) != 0) {
        std::cerr << "[ComputePipeline] Invalid SPIR-V file size: " << path << std::endl;
        return {};
    }

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    // Validate SPIR-V magic number
    if (!buffer.empty() && buffer[0] != 0x07230203) {
        std::cerr << "[ComputePipeline] Invalid SPIR-V magic number in: " << path << std::endl;
        return {};
    }

    return buffer;
}

vk::raii::ShaderModule ComputePipelineManager::createShaderModule(
    const std::vector<uint32_t>& spirvCode
) {
    vk::ShaderModuleCreateInfo createInfo{
        {},
        spirvCode.size() * sizeof(uint32_t),
        spirvCode.data()
    };

    return vk::raii::ShaderModule(_device, createInfo);
}

// SubgroupUtils implementations

namespace SubgroupUtils {

uint32_t getRecommendedWorkgroupSize(
    const SubgroupCapabilities& caps,
    uint32_t problemSize,
    uint32_t maxWorkgroupSize
) {
    if (caps.subgroupSize == 0) {
        // No subgroup info, use default
        return std::min(problemSize, maxWorkgroupSize);
    }

    // Calculate optimal workgroup size
    // - Should be multiple of subgroup size
    // - Should not exceed max workgroup size
    // - Should be reasonable for the problem size

    uint32_t idealSize = caps.subgroupSize * 4;  // Typical sweet spot

    // Clamp to max workgroup size
    idealSize = std::min(idealSize, maxWorkgroupSize);

    // Align to subgroup size
    idealSize = alignToSubgroup(idealSize, caps.subgroupSize);

    // For small problems, use smaller workgroups
    if (problemSize < idealSize) {
        idealSize = alignToSubgroup(problemSize, caps.subgroupSize);
        if (idealSize == 0) idealSize = caps.subgroupSize;
    }

    return idealSize;
}

bool shaderUsesSubgroupOps(const std::vector<uint32_t>& spirvCode) {
    if (spirvCode.size() < 5) {
        return false;
    }

    // Simple heuristic: search for subgroup-related SPIR-V opcodes
    // OpGroupNonUniformElect = 333
    // OpGroupNonUniformAll = 334
    // OpGroupNonUniformAny = 335
    // OpGroupNonUniformBroadcast = 337
    // OpSubgroupReadInvocationKHR = 4432
    // OpSubgroupBallotKHR = 4421

    static const std::vector<uint32_t> subgroupOpcodes = {
        333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348,
        4421, 4422, 4428, 4429, 4430, 4431, 4432
    };

    // Skip header (5 words)
    for (size_t i = 5; i < spirvCode.size(); ) {
        uint32_t instruction = spirvCode[i];
        uint16_t opcode = instruction & 0xFFFF;
        uint16_t wordCount = instruction >> 16;

        if (wordCount == 0) break;  // Malformed

        // Check if this is a subgroup opcode
        for (uint32_t subOp : subgroupOpcodes) {
            if (opcode == subOp) {
                return true;
            }
        }

        i += wordCount;
    }

    return false;
}

} // namespace SubgroupUtils

} // namespace mlsdk
