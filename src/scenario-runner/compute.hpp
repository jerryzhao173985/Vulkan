/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compute Pipeline with Subgroup Operation Support
 * Provides advanced Vulkan 1.4 compute features with fallback paths
 */

#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <memory>
#include <vector>
#include <optional>
#include <string>
#include <functional>
#include <unordered_map>

namespace mlsdk {

/**
 * Subgroup feature capabilities detected from the physical device
 */
struct SubgroupCapabilities {
    // Core subgroup properties
    uint32_t subgroupSize = 0;
    VkShaderStageFlags supportedStages = 0;
    VkSubgroupFeatureFlags supportedOperations = 0;
    bool quadOperationsInAllStages = false;

    // Feature flags for specific operations
    bool hasBasic = false;
    bool hasVote = false;
    bool hasArithmetic = false;
    bool hasBallot = false;
    bool hasShuffle = false;
    bool hasShuffleRelative = false;
    bool hasClustered = false;
    bool hasQuad = false;

    // Advanced features (Vulkan 1.1+)
    bool hasPartitionedNV = false;

    // Fallback mode when subgroup operations not available
    bool requiresFallback = false;

    /**
     * Check if a specific subgroup operation is supported
     */
    bool supportsOperation(VkSubgroupFeatureFlagBits operation) const {
        return (supportedOperations & operation) != 0;
    }

    /**
     * Check if compute stage supports subgroup operations
     */
    bool supportsComputeStage() const {
        return (supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    }

    /**
     * Get optimal workgroup size based on subgroup size
     */
    uint32_t getOptimalWorkgroupSize(uint32_t desiredSize = 256) const {
        if (subgroupSize == 0) return desiredSize;
        // Align to subgroup size for optimal performance
        return ((desiredSize + subgroupSize - 1) / subgroupSize) * subgroupSize;
    }
};

/**
 * Compute pipeline configuration
 */
struct ComputePipelineConfig {
    std::string shaderPath;
    std::string entryPoint = "main";
    uint32_t workgroupSizeX = 64;
    uint32_t workgroupSizeY = 1;
    uint32_t workgroupSizeZ = 1;
    bool useSubgroupOperations = true;
    bool enablePipelineCache = true;
    std::vector<vk::PushConstantRange> pushConstantRanges;
    std::vector<vk::DescriptorSetLayoutBinding> descriptorBindings;
};

/**
 * Dispatch configuration for compute operations
 */
struct DispatchConfig {
    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
    bool indirect = false;
    vk::Buffer indirectBuffer = VK_NULL_HANDLE;
    vk::DeviceSize indirectOffset = 0;
};

/**
 * Compute Pipeline Manager
 * Handles compute pipeline creation with subgroup operation support and fallback paths
 */
class ComputePipelineManager {
public:
    ComputePipelineManager(
        vk::raii::Device& device,
        vk::raii::PhysicalDevice& physicalDevice,
        uint32_t computeQueueFamilyIndex
    );

    ~ComputePipelineManager();

    // Non-copyable
    ComputePipelineManager(const ComputePipelineManager&) = delete;
    ComputePipelineManager& operator=(const ComputePipelineManager&) = delete;

    // Move operations
    ComputePipelineManager(ComputePipelineManager&&) noexcept;
    ComputePipelineManager& operator=(ComputePipelineManager&&) noexcept;

    /**
     * Query and cache subgroup capabilities from physical device
     */
    void querySubgroupCapabilities();

    /**
     * Get the detected subgroup capabilities
     */
    const SubgroupCapabilities& getSubgroupCapabilities() const { return _subgroupCaps; }

    /**
     * Check if subgroup operations are available (not requiring fallback)
     */
    bool hasSubgroupSupport() const { return !_subgroupCaps.requiresFallback; }

    /**
     * Create a compute pipeline with automatic subgroup optimization
     * Falls back to standard implementation if subgroups not supported
     */
    bool createPipeline(const ComputePipelineConfig& config, const std::string& pipelineName);

    /**
     * Create a pipeline using subgroup-optimized shader if available,
     * otherwise use the fallback shader
     */
    bool createPipelineWithFallback(
        const std::string& subgroupShaderPath,
        const std::string& fallbackShaderPath,
        const ComputePipelineConfig& baseConfig,
        const std::string& pipelineName
    );

    /**
     * Get a created pipeline by name
     */
    vk::Pipeline getPipeline(const std::string& name) const;

    /**
     * Get pipeline layout for a named pipeline
     */
    vk::PipelineLayout getPipelineLayout(const std::string& name) const;

    /**
     * Record dispatch commands with appropriate barriers
     */
    void recordDispatch(
        vk::raii::CommandBuffer& cmdBuffer,
        const std::string& pipelineName,
        const DispatchConfig& dispatch,
        const void* pushConstantData = nullptr,
        uint32_t pushConstantSize = 0
    );

    /**
     * Record dispatch with subgroup-aware barrier handling
     */
    void recordSubgroupDispatch(
        vk::raii::CommandBuffer& cmdBuffer,
        const std::string& pipelineName,
        const DispatchConfig& dispatch,
        bool requiresSubgroupBarrier = false
    );

    /**
     * Get specialization constants for subgroup-aware shaders
     */
    std::vector<vk::SpecializationMapEntry> getSubgroupSpecializationEntries() const;

    /**
     * Create specialization data for subgroup parameters
     */
    struct SubgroupSpecializationData {
        uint32_t subgroupSize;
        uint32_t useSubgroupOps;
        uint32_t workgroupSize;
    };

    SubgroupSpecializationData createSubgroupSpecData(uint32_t workgroupSize = 256) const;

    /**
     * Clear all cached pipelines
     */
    void clearPipelines();

    /**
     * Get pipeline cache for serialization
     */
    vk::PipelineCache getPipelineCache() const;

private:
    vk::raii::Device& _device;
    vk::raii::PhysicalDevice& _physicalDevice;
    uint32_t _computeQueueFamilyIndex;

    SubgroupCapabilities _subgroupCaps;

    // Pipeline cache for faster pipeline creation
    std::unique_ptr<vk::raii::PipelineCache> _pipelineCache;

    // Named pipelines storage
    struct PipelineEntry {
        std::unique_ptr<vk::raii::Pipeline> pipeline;
        std::unique_ptr<vk::raii::PipelineLayout> layout;
        std::unique_ptr<vk::raii::DescriptorSetLayout> descriptorSetLayout;
        bool usesSubgroups;
    };
    std::unordered_map<std::string, PipelineEntry> _pipelines;

    /**
     * Load SPIR-V shader from file
     */
    std::vector<uint32_t> loadShaderSPIRV(const std::string& path);

    /**
     * Create shader module from SPIR-V code
     */
    vk::raii::ShaderModule createShaderModule(const std::vector<uint32_t>& spirvCode);

    /**
     * Detect if hardware supports the minimum required subgroup features
     */
    bool detectMinimumSubgroupSupport() const;

    /**
     * Initialize pipeline cache
     */
    void initializePipelineCache();
};

/**
 * Utility functions for subgroup-optimized compute operations
 */
namespace SubgroupUtils {

/**
 * Calculate optimal dispatch size aligned to subgroup size
 */
inline uint32_t alignToSubgroup(uint32_t value, uint32_t subgroupSize) {
    if (subgroupSize == 0) return value;
    return ((value + subgroupSize - 1) / subgroupSize) * subgroupSize;
}

/**
 * Calculate number of workgroups for a given problem size
 */
inline uint32_t calculateWorkgroupCount(uint32_t problemSize, uint32_t workgroupSize) {
    return (problemSize + workgroupSize - 1) / workgroupSize;
}

/**
 * Get recommended workgroup size based on subgroup properties
 */
uint32_t getRecommendedWorkgroupSize(
    const SubgroupCapabilities& caps,
    uint32_t problemSize,
    uint32_t maxWorkgroupSize = 256
);

/**
 * Check if a shader requires subgroup operations based on SPIR-V analysis
 * (Simple heuristic check)
 */
bool shaderUsesSubgroupOps(const std::vector<uint32_t>& spirvCode);

} // namespace SubgroupUtils

} // namespace mlsdk
