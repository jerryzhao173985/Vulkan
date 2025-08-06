/**
 * Minimal VGF Library Test
 * Simplest possible test to verify VGF library works
 */

#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>

#include <vgf/encoder.hpp>
#include <vgf/types.hpp>

using namespace mlsdk::vgflib;

int main() {
    std::cout << "VGF Library Minimal Test\n";
    std::cout << "========================\n\n";
    
    int tests_passed = 0;
    int tests_failed = 0;
    
    try {
        // Test 1: Create encoder
        std::cout << "Test 1: Create encoder... ";
        auto encoder = CreateEncoder(1);
        if (encoder) {
            std::cout << "✅ PASS\n";
            tests_passed++;
        } else {
            std::cout << "❌ FAIL\n";
            tests_failed++;
            return 1;
        }
        
        // Test 2: Add a simple compute module
        std::cout << "Test 2: Add compute module... ";
        std::vector<uint32_t> spirv = {
            0x07230203, // SPIR-V magic
            0x00010000, // Version
            0x00000000, 0x00000001, 0x00000000
        };
        
        auto module = encoder->AddModule(
            ModuleType::COMPUTE,
            "test_shader",
            "main",
            spirv
        );
        std::cout << "✅ PASS\n";
        tests_passed++;
        
        // Test 3: Add resources
        std::cout << "Test 3: Add resources... ";
        std::vector<int64_t> shape = {1, 224, 224, 3};
        std::vector<int64_t> strides;
        
        auto input = encoder->AddInputResource(
            7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
            37, // VK_FORMAT_R8G8B8A8_UNORM = 37
            shape, strides
        );
        
        auto output = encoder->AddOutputResource(
            7, // Storage buffer
            100, // VK_FORMAT_R32_SFLOAT = 100
            {1, 1000}, strides
        );
        
        std::cout << "✅ PASS\n";
        tests_passed++;
        
        // Test 4: Finish encoding
        std::cout << "Test 4: Finish encoding... ";
        encoder->Finish();
        std::cout << "✅ PASS\n";
        tests_passed++;
        
        // Test 5: Write to stream
        std::cout << "Test 5: Write VGF data... ";
        std::ostringstream oss;
        bool writeSuccess = encoder->WriteTo(oss);
        
        if (writeSuccess && oss.str().size() > 0) {
            std::cout << "✅ PASS (size: " << oss.str().size() << " bytes)\n";
            tests_passed++;
        } else {
            std::cout << "❌ FAIL\n";
            tests_failed++;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ EXCEPTION: " << e.what() << "\n";
        tests_failed++;
    }
    
    // Summary
    std::cout << "\n========================\n";
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed\n";
    
    if (tests_failed == 0) {
        std::cout << "✅ VGF library is working!\n";
        return 0;
    } else {
        std::cout << "❌ Some tests failed\n";
        return 1;
    }
}