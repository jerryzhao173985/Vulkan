/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 *
 * Vulkan Context Implementation with Cooperative Matrix Support
 * Provides VK_KHR_cooperative_matrix query and automatic fallback detection
 */

#include "context.hpp"

#include <iostream>
#include <algorithm>
#include <cstring>

namespace mlsdk {

// Extension name constant
static constexpr const char* VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME = "VK_KHR_cooperative_matrix";

VulkanContext::VulkanContext(
    vk::raii::Instance& instance,
    vk::raii::PhysicalDevice& physicalDevice
) : _instance(instance)
  , _physicalDevice(physicalDevice)
{
    // Cache basic device properties
    _deviceProperties = _physicalDevice.getProperties();
    _deviceFeatures = _physicalDevice.getFeatures();

    // Cache available extensions
    cacheAvailableExtensions();

    // Query cooperative matrix capabilities
    queryCooperativeMatrixCapabilities();
}

VulkanContext::~VulkanContext() = default;

VulkanContext::VulkanContext(VulkanContext&& other) noexcept
    : _instance(other._instance)
    , _physicalDevice(other._physicalDevice)
    , _deviceProperties(std::move(other._deviceProperties))
    , _deviceFeatures(std::move(other._deviceFeatures))
    , _availableExtensions(std::move(other._availableExtensions))
    , _coopMatrixCaps(std::move(other._coopMatrixCaps))
{
}

VulkanContext& VulkanContext::operator=(VulkanContext&& other) noexcept {
    if (this != &other) {
        _deviceProperties = std::move(other._deviceProperties);
        _deviceFeatures = std::move(other._deviceFeatures);
        _availableExtensions = std::move(other._availableExtensions);
        _coopMatrixCaps = std::move(other._coopMatrixCaps);
    }
    return *this;
}

void VulkanContext::cacheAvailableExtensions() {
    _availableExtensions = _physicalDevice.enumerateDeviceExtensionProperties();
}

bool VulkanContext::isExtensionSupported(const std::string& extensionName) const {
    for (const auto& ext : _availableExtensions) {
        if (extensionName == ext.extensionName) {
            return true;
        }
    }
    return false;
}

bool VulkanContext::hasCooperativeMatrixExtension() const {
    return isExtensionSupported(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
}

void VulkanContext::queryCooperativeMatrixCapabilities() {
    // Check if extension is available
    _coopMatrixCaps.extensionSupported = hasCooperativeMatrixExtension();

    if (!_coopMatrixCaps.extensionSupported) {
        _coopMatrixCaps.requiresFallback = true;
        std::cout << "[VulkanContext] VK_KHR_cooperative_matrix extension not available" << std::endl;
        std::cout << "[VulkanContext] Cooperative matrix operations will use fallback path" << std::endl;
        return;
    }

    // Query cooperative matrix features
    queryCooperativeMatrixFeatures();

    // Query supported configurations
    queryCooperativeMatrixProperties();

    // Determine if fallback is needed
    _coopMatrixCaps.requiresFallback = !detectMinimumCooperativeMatrixSupport();

    // Find best ML configuration
    if (!_coopMatrixCaps.requiresFallback) {
        findBestMLConfiguration();
    }

    // Log detected capabilities
    logCapabilities();
}

void VulkanContext::queryCooperativeMatrixFeatures() {
    // Query cooperative matrix features using Vulkan 1.1+ feature chaining
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrixFeatures{};
    coopMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopMatrixFeatures.pNext = nullptr;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &coopMatrixFeatures;

    vkGetPhysicalDeviceFeatures2(*_physicalDevice, &features2);

    // Store feature flags
    _coopMatrixCaps.cooperativeMatrix = coopMatrixFeatures.cooperativeMatrix == VK_TRUE;
    _coopMatrixCaps.cooperativeMatrixRobustBufferAccess =
        coopMatrixFeatures.cooperativeMatrixRobustBufferAccess == VK_TRUE;
}

void VulkanContext::queryCooperativeMatrixProperties() {
    // Query supported cooperative matrix properties
    // First, get the count of supported configurations
    uint32_t propertyCount = 0;

    // Use the Vulkan function to enumerate cooperative matrix properties
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR vkGetCoopMatrixProps =
        reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            _instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")
        );

    if (!vkGetCoopMatrixProps) {
        std::cout << "[VulkanContext] Failed to load vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR" << std::endl;
        return;
    }

    VkResult result = vkGetCoopMatrixProps(*_physicalDevice, &propertyCount, nullptr);
    if (result != VK_SUCCESS || propertyCount == 0) {
        std::cout << "[VulkanContext] No cooperative matrix configurations available" << std::endl;
        return;
    }

    // Allocate and query the properties
    std::vector<VkCooperativeMatrixPropertiesKHR> properties(propertyCount);
    for (auto& prop : properties) {
        prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        prop.pNext = nullptr;
    }

    result = vkGetCoopMatrixProps(*_physicalDevice, &propertyCount, properties.data());
    if (result != VK_SUCCESS) {
        std::cout << "[VulkanContext] Failed to query cooperative matrix properties" << std::endl;
        return;
    }

    // Convert to our configuration format
    _coopMatrixCaps.supportedConfigs.reserve(propertyCount);

    for (const auto& prop : properties) {
        CooperativeMatrixConfig config;
        config.MSize = prop.MSize;
        config.NSize = prop.NSize;
        config.KSize = prop.KSize;
        config.AType = prop.AType;
        config.BType = prop.BType;
        config.CType = prop.CType;
        config.ResultType = prop.ResultType;
        config.saturatingAccumulation = prop.saturatingAccumulation == VK_TRUE;
        config.scope = prop.scope;

        _coopMatrixCaps.supportedConfigs.push_back(config);
    }
}

bool VulkanContext::detectMinimumCooperativeMatrixSupport() const {
    // Minimum requirements for cooperative matrix optimization:
    // 1. Extension is supported
    // 2. cooperativeMatrix feature is enabled
    // 3. At least one matrix configuration is available
    return _coopMatrixCaps.extensionSupported &&
           _coopMatrixCaps.cooperativeMatrix &&
           !_coopMatrixCaps.supportedConfigs.empty();
}

void VulkanContext::findBestMLConfiguration() {
    // Find the best configuration for ML workloads
    // Priority: fp16 input with fp32 accumulator, largest matrix size

    size_t bestIndex = SIZE_MAX;
    uint32_t bestScore = 0;

    for (size_t i = 0; i < _coopMatrixCaps.supportedConfigs.size(); ++i) {
        const auto& config = _coopMatrixCaps.supportedConfigs[i];

        // Score based on ML suitability
        uint32_t score = 0;

        // Prefer fp16 input (common for ML inference)
        if (config.supportsFloat16Input()) {
            score += 1000;
        }

        // Prefer fp32 accumulator (for numerical stability)
        if (config.supportsFloat32Accumulator()) {
            score += 500;
        }

        // Prefer larger matrix sizes for throughput
        score += config.MSize * config.NSize;

        // Prefer subgroup scope (most common)
        if (config.scope == VK_SCOPE_SUBGROUP_KHR) {
            score += 100;
        }

        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }

    if (bestIndex != SIZE_MAX) {
        _coopMatrixCaps.bestMLConfigIndex = bestIndex;
    }
}

void VulkanContext::logCapabilities() const {
    std::cout << "[VulkanContext] Cooperative matrix capabilities detected:" << std::endl;
    std::cout << "  Extension supported: " << (_coopMatrixCaps.extensionSupported ? "YES" : "NO") << std::endl;

    if (_coopMatrixCaps.extensionSupported) {
        std::cout << "  cooperativeMatrix feature: "
                  << (_coopMatrixCaps.cooperativeMatrix ? "YES" : "NO") << std::endl;
        std::cout << "  Robust buffer access: "
                  << (_coopMatrixCaps.cooperativeMatrixRobustBufferAccess ? "YES" : "NO") << std::endl;
        std::cout << "  Supported configurations: " << _coopMatrixCaps.supportedConfigs.size() << std::endl;

        // Log up to 5 configurations
        size_t logCount = std::min<size_t>(_coopMatrixCaps.supportedConfigs.size(), 5);
        for (size_t i = 0; i < logCount; ++i) {
            const auto& config = _coopMatrixCaps.supportedConfigs[i];
            std::cout << "    [" << i << "] M=" << config.MSize
                      << " N=" << config.NSize
                      << " K=" << config.KSize
                      << " A=" << componentTypeToString(config.AType)
                      << " B=" << componentTypeToString(config.BType)
                      << " C=" << componentTypeToString(config.CType)
                      << " scope=" << scopeToString(config.scope)
                      << std::endl;
        }

        if (_coopMatrixCaps.supportedConfigs.size() > 5) {
            std::cout << "    ... and " << (_coopMatrixCaps.supportedConfigs.size() - 5)
                      << " more configurations" << std::endl;
        }

        if (_coopMatrixCaps.bestMLConfigIndex) {
            const auto* bestConfig = _coopMatrixCaps.getBestMLConfig();
            if (bestConfig) {
                std::cout << "  Best ML config: M=" << bestConfig->MSize
                          << " N=" << bestConfig->NSize
                          << " K=" << bestConfig->KSize
                          << std::endl;
            }
        }
    }

    std::cout << "  Requires fallback path: " << (_coopMatrixCaps.requiresFallback ? "YES" : "NO") << std::endl;
}

std::vector<const char*> VulkanContext::getRequiredExtensions() const {
    std::vector<const char*> extensions;
    // Add required extensions for basic operation
    // (cooperative matrix is optional)
    return extensions;
}

std::vector<const char*> VulkanContext::getOptionalExtensions() const {
    std::vector<const char*> extensions;
    if (_coopMatrixCaps.extensionSupported) {
        extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
    return extensions;
}

std::optional<CooperativeMatrixConfig> VulkanContext::selectOptimalConfig(
    uint32_t preferredM,
    uint32_t preferredN,
    uint32_t preferredK,
    VkComponentTypeKHR preferredType
) const {
    if (_coopMatrixCaps.supportedConfigs.empty()) {
        return std::nullopt;
    }

    // Try to find exact match first
    for (const auto& config : _coopMatrixCaps.supportedConfigs) {
        if (config.MSize == preferredM &&
            config.NSize == preferredN &&
            config.KSize == preferredK &&
            config.AType == preferredType) {
            return config;
        }
    }

    // Try to find config with matching type
    for (const auto& config : _coopMatrixCaps.supportedConfigs) {
        if (config.AType == preferredType) {
            return config;
        }
    }

    // Return best ML config as fallback
    if (const auto* bestConfig = _coopMatrixCaps.getBestMLConfig()) {
        return *bestConfig;
    }

    // Return first available config
    if (!_coopMatrixCaps.supportedConfigs.empty()) {
        return _coopMatrixCaps.supportedConfigs[0];
    }

    return std::nullopt;
}

std::string VulkanContext::componentTypeToString(VkComponentTypeKHR type) {
    switch (type) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "fp16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "fp32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "fp64";
        case VK_COMPONENT_TYPE_SINT8_KHR: return "i8";
        case VK_COMPONENT_TYPE_SINT16_KHR: return "i16";
        case VK_COMPONENT_TYPE_SINT32_KHR: return "i32";
        case VK_COMPONENT_TYPE_SINT64_KHR: return "i64";
        case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
        case VK_COMPONENT_TYPE_UINT16_KHR: return "u16";
        case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
        case VK_COMPONENT_TYPE_UINT64_KHR: return "u64";
        default: return "unknown";
    }
}

std::string VulkanContext::scopeToString(VkScopeKHR scope) {
    switch (scope) {
        case VK_SCOPE_DEVICE_KHR: return "device";
        case VK_SCOPE_WORKGROUP_KHR: return "workgroup";
        case VK_SCOPE_SUBGROUP_KHR: return "subgroup";
        case VK_SCOPE_QUEUE_FAMILY_KHR: return "queue_family";
        default: return "unknown";
    }
}

// CooperativeMatrixUtils implementations

namespace CooperativeMatrixUtils {

uint32_t calculateOptimalWorkgroupSize(
    const CooperativeMatrixCapabilities& caps,
    uint32_t matrixM,
    uint32_t matrixN
) {
    if (!caps.isAvailable() || caps.supportedConfigs.empty()) {
        // Fallback: use standard tile size
        return getFallbackTileSize();
    }

    // Find a configuration that fits the matrix dimensions
    for (const auto& config : caps.supportedConfigs) {
        if (matrixM >= config.MSize && matrixN >= config.NSize) {
            // Workgroup size should be multiple of cooperative matrix tile
            return config.MSize * config.NSize;
        }
    }

    // Use best ML config if available
    if (const auto* bestConfig = caps.getBestMLConfig()) {
        return bestConfig->MSize * bestConfig->NSize;
    }

    return getFallbackTileSize();
}

bool shaderUsesCooperativeMatrix(const std::vector<uint32_t>& spirvCode) {
    if (spirvCode.size() < 5) {
        return false;
    }

    // Check for cooperative matrix related SPIR-V opcodes
    // OpTypeCooperativeMatrixKHR = 4456
    // OpCooperativeMatrixLoadKHR = 4457
    // OpCooperativeMatrixStoreKHR = 4458
    // OpCooperativeMatrixMulAddKHR = 4459
    // OpCooperativeMatrixLengthKHR = 4460
    // Also check for capability: CooperativeMatrixKHR = 6022

    static const std::vector<uint32_t> coopMatrixOpcodes = {
        4456, 4457, 4458, 4459, 4460
    };

    static const uint32_t coopMatrixCapability = 6022;

    // Skip SPIR-V header (5 words)
    for (size_t i = 5; i < spirvCode.size(); ) {
        uint32_t instruction = spirvCode[i];
        uint16_t opcode = instruction & 0xFFFF;
        uint16_t wordCount = instruction >> 16;

        if (wordCount == 0) break;  // Malformed SPIR-V

        // Check for OpCapability with CooperativeMatrixKHR
        if (opcode == 17) {  // OpCapability
            if (i + 1 < spirvCode.size() && spirvCode[i + 1] == coopMatrixCapability) {
                return true;
            }
        }

        // Check for cooperative matrix opcodes
        for (uint32_t coopOp : coopMatrixOpcodes) {
            if (opcode == coopOp) {
                return true;
            }
        }

        i += wordCount;
    }

    return false;
}

std::string configToString(const CooperativeMatrixConfig& config) {
    std::string result = "CoopMat[M=" + std::to_string(config.MSize);
    result += " N=" + std::to_string(config.NSize);
    result += " K=" + std::to_string(config.KSize);
    result += " A=" + VulkanContext::componentTypeToString(config.AType);
    result += " B=" + VulkanContext::componentTypeToString(config.BType);
    result += " C=" + VulkanContext::componentTypeToString(config.CType);
    result += "]";
    return result;
}

} // namespace CooperativeMatrixUtils

} // namespace mlsdk
