/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 *
 * Vulkan Context with Cooperative Matrix Support
 * Provides VK_KHR_cooperative_matrix query and fallback detection
 */

#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <memory>
#include <vector>
#include <optional>
#include <string>

namespace mlsdk {

/**
 * Cooperative matrix type configuration
 * Represents a supported matrix configuration for the device
 */
struct CooperativeMatrixConfig {
    uint32_t MSize = 0;      // M dimension
    uint32_t NSize = 0;      // N dimension
    uint32_t KSize = 0;      // K dimension
    VkComponentTypeKHR AType = VK_COMPONENT_TYPE_MAX_ENUM_KHR;
    VkComponentTypeKHR BType = VK_COMPONENT_TYPE_MAX_ENUM_KHR;
    VkComponentTypeKHR CType = VK_COMPONENT_TYPE_MAX_ENUM_KHR;
    VkComponentTypeKHR ResultType = VK_COMPONENT_TYPE_MAX_ENUM_KHR;
    bool saturatingAccumulation = false;
    VkScopeKHR scope = VK_SCOPE_SUBGROUP_KHR;

    /**
     * Check if this config supports float16 A/B matrices (common for ML)
     */
    bool supportsFloat16Input() const {
        return AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
               BType == VK_COMPONENT_TYPE_FLOAT16_KHR;
    }

    /**
     * Check if this config supports float32 accumulator
     */
    bool supportsFloat32Accumulator() const {
        return CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
               ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR;
    }

    /**
     * Get total elements in this matrix config (M * N * K for GEMM)
     */
    uint32_t getTotalElements() const {
        return MSize * NSize * KSize;
    }
};

/**
 * Cooperative matrix capabilities detected from the physical device
 */
struct CooperativeMatrixCapabilities {
    // Extension availability
    bool extensionSupported = false;
    bool extensionEnabled = false;

    // Feature flags from VkPhysicalDeviceCooperativeMatrixFeaturesKHR
    bool cooperativeMatrix = false;
    bool cooperativeMatrixRobustBufferAccess = false;

    // Supported matrix configurations
    std::vector<CooperativeMatrixConfig> supportedConfigs;

    // Fallback mode when cooperative matrix not available
    bool requiresFallback = false;

    // Best configuration for ML workloads (fp16 input, fp32 accumulator)
    std::optional<size_t> bestMLConfigIndex;

    /**
     * Check if cooperative matrix operations are available
     */
    bool isAvailable() const {
        return extensionSupported && cooperativeMatrix && !requiresFallback;
    }

    /**
     * Check if a specific matrix size is supported
     */
    bool supportsMNK(uint32_t M, uint32_t N, uint32_t K) const {
        for (const auto& config : supportedConfigs) {
            if (config.MSize == M && config.NSize == N && config.KSize == K) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get the best configuration for ML workloads
     * Returns nullptr if no suitable configuration found
     */
    const CooperativeMatrixConfig* getBestMLConfig() const {
        if (bestMLConfigIndex && *bestMLConfigIndex < supportedConfigs.size()) {
            return &supportedConfigs[*bestMLConfigIndex];
        }
        return nullptr;
    }

    /**
     * Find configurations matching specific component types
     */
    std::vector<const CooperativeMatrixConfig*> findConfigs(
        VkComponentTypeKHR aType,
        VkComponentTypeKHR bType,
        VkComponentTypeKHR cType
    ) const {
        std::vector<const CooperativeMatrixConfig*> matches;
        for (const auto& config : supportedConfigs) {
            if (config.AType == aType && config.BType == bType && config.CType == cType) {
                matches.push_back(&config);
            }
        }
        return matches;
    }
};

/**
 * Vulkan Context Manager
 * Handles device context initialization with cooperative matrix support
 * and automatic fallback detection for unsupported hardware
 */
class VulkanContext {
public:
    VulkanContext(
        vk::raii::Instance& instance,
        vk::raii::PhysicalDevice& physicalDevice
    );

    ~VulkanContext();

    // Non-copyable
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // Move operations
    VulkanContext(VulkanContext&&) noexcept;
    VulkanContext& operator=(VulkanContext&&) noexcept;

    /**
     * Query and cache cooperative matrix capabilities from physical device
     */
    void queryCooperativeMatrixCapabilities();

    /**
     * Check if VK_KHR_cooperative_matrix extension is available
     */
    bool hasCooperativeMatrixExtension() const;

    /**
     * Get the detected cooperative matrix capabilities
     */
    const CooperativeMatrixCapabilities& getCooperativeMatrixCapabilities() const {
        return _coopMatrixCaps;
    }

    /**
     * Check if cooperative matrix operations are available (not requiring fallback)
     */
    bool hasCooperativeMatrixSupport() const {
        return _coopMatrixCaps.isAvailable();
    }

    /**
     * Check if fallback is required for matrix operations
     */
    bool requiresCooperativeMatrixFallback() const {
        return _coopMatrixCaps.requiresFallback;
    }

    /**
     * Get the list of required device extensions for cooperative matrix
     */
    std::vector<const char*> getRequiredExtensions() const;

    /**
     * Get the list of optional extensions (includes cooperative matrix)
     */
    std::vector<const char*> getOptionalExtensions() const;

    /**
     * Check if a specific device extension is supported
     */
    bool isExtensionSupported(const std::string& extensionName) const;

    /**
     * Get the physical device properties
     */
    const vk::PhysicalDeviceProperties& getPhysicalDeviceProperties() const {
        return _deviceProperties;
    }

    /**
     * Get the physical device features
     */
    const vk::PhysicalDeviceFeatures& getPhysicalDeviceFeatures() const {
        return _deviceFeatures;
    }

    /**
     * Log detected capabilities to console
     */
    void logCapabilities() const;

    /**
     * Select optimal matrix configuration for a given workload
     * Returns empty optional if no suitable config found
     */
    std::optional<CooperativeMatrixConfig> selectOptimalConfig(
        uint32_t preferredM,
        uint32_t preferredN,
        uint32_t preferredK,
        VkComponentTypeKHR preferredType = VK_COMPONENT_TYPE_FLOAT16_KHR
    ) const;

private:
    vk::raii::Instance& _instance;
    vk::raii::PhysicalDevice& _physicalDevice;

    // Cached device properties
    vk::PhysicalDeviceProperties _deviceProperties;
    vk::PhysicalDeviceFeatures _deviceFeatures;

    // Extension support cache
    std::vector<vk::ExtensionProperties> _availableExtensions;

    // Cooperative matrix capabilities
    CooperativeMatrixCapabilities _coopMatrixCaps;

    /**
     * Initialize extension support cache
     */
    void cacheAvailableExtensions();

    /**
     * Query cooperative matrix features structure
     */
    void queryCooperativeMatrixFeatures();

    /**
     * Query supported cooperative matrix configurations
     */
    void queryCooperativeMatrixProperties();

    /**
     * Detect if hardware meets minimum requirements for cooperative matrix
     */
    bool detectMinimumCooperativeMatrixSupport() const;

    /**
     * Find the best configuration for ML workloads
     */
    void findBestMLConfiguration();

public:
    /**
     * Convert VkComponentTypeKHR to string for logging
     */
    static std::string componentTypeToString(VkComponentTypeKHR type);

    /**
     * Convert VkScopeKHR to string for logging
     */
    static std::string scopeToString(VkScopeKHR scope);
};

/**
 * Utility functions for cooperative matrix operations
 */
namespace CooperativeMatrixUtils {

/**
 * Calculate optimal workgroup size for cooperative matrix operations
 */
uint32_t calculateOptimalWorkgroupSize(
    const CooperativeMatrixCapabilities& caps,
    uint32_t matrixM,
    uint32_t matrixN
);

/**
 * Check if a SPIR-V shader uses cooperative matrix operations
 */
bool shaderUsesCooperativeMatrix(const std::vector<uint32_t>& spirvCode);

/**
 * Get fallback tile size when cooperative matrix is not available
 */
inline uint32_t getFallbackTileSize() {
    return 16;  // Standard tile size for non-cooperative matrix path
}

/**
 * Get string representation of cooperative matrix configuration
 */
std::string configToString(const CooperativeMatrixConfig& config);

} // namespace CooperativeMatrixUtils

} // namespace mlsdk
