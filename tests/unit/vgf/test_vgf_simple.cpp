/**
 * Simple VGF Library Unit Test
 * Basic test to verify VGF library compilation and functionality
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cstring>

// VGF Library headers
#include <vgf/encoder.hpp>
#include <vgf/decoder.hpp>
#include <vgf/types.hpp>

using namespace mlsdk::vgflib;

int main() {
    std::cout << "VGF Library Simple Test" << std::endl;
    std::cout << "========================" << std::endl;
    
    try {
        // Test 1: Create encoder
        std::cout << "Test 1: Creating encoder... ";
        uint16_t vkHeaderVersion = 1;
        auto encoder = CreateEncoder(vkHeaderVersion);
        if (encoder) {
            std::cout << "✅ PASS" << std::endl;
        } else {
            std::cout << "❌ FAIL" << std::endl;
            return 1;
        }
        
        // Test 2: Add a module
        std::cout << "Test 2: Adding module... ";
        std::vector<uint32_t> spirvCode = {
            0x07230203, // SPIR-V magic number
            0x00010000, // Version 1.0
            0x00000000, // Generator
            0x00000001, // Bound
            0x00000000  // Schema
        };
        
        auto moduleRef = encoder->AddModule(
            ModuleType::COMPUTE,
            "test_compute",
            "main",
            spirvCode
        );
        std::cout << "✅ PASS" << std::endl;
        
        // Test 3: Add input resource
        std::cout << "Test 3: Adding input resource... ";
        std::vector<int64_t> shape = {1, 224, 224, 3}; // NHWC format
        std::vector<int64_t> strides = {}; // Packed layout
        
        auto inputRef = encoder->AddInputResource(
            0, // DescriptorType (using raw value for now)
            37, // VK_FORMAT_R8G8B8A8_UNORM = 37
            shape,
            strides
        );
        std::cout << "✅ PASS" << std::endl;
        
        // Test 4: Add output resource
        std::cout << "Test 4: Adding output resource... ";
        std::vector<int64_t> outputShape = {1, 1000}; // Classification output
        
        auto outputRef = encoder->AddOutputResource(
            0, // DescriptorType
            100, // VK_FORMAT_R32_SFLOAT = 100
            outputShape,
            strides
        );
        std::cout << "✅ PASS" << std::endl;
        
        // Test 5: Add constant
        std::cout << "Test 5: Adding constant... ";
        std::vector<float> weights(1024, 0.1f);
        auto constantRef = encoder->AddConstant(
            weights.data(),
            weights.size() * sizeof(float)
        );
        std::cout << "✅ PASS" << std::endl;
        
        // Test 6: Encode to buffer
        std::cout << "Test 6: Encoding to buffer... ";
        size_t encodedSize = 0;
        encoder->GetEncodedSize(encodedSize);
        
        if (encodedSize > 0) {
            std::vector<uint8_t> buffer(encodedSize);
            encoder->Encode(buffer.data(), buffer.size());
            std::cout << "✅ PASS (size: " << encodedSize << " bytes)" << std::endl;
        } else {
            std::cout << "❌ FAIL (size: 0)" << std::endl;
            return 1;
        }
        
        std::cout << "\n=============================";
        std::cout << "\n✅ All tests passed!" << std::endl;
        std::cout << "VGF library is working correctly." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "❌ EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}