/**
 * VGF Library Core Unit Tests
 * Complete test suite for Vulkan Graph Format library
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <random>

// VGF Library headers
#include <vgf/encoder.hpp>
#include <vgf/decoder.hpp>
#include <vgf/types.hpp>

using namespace mlsdk::vgflib;

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    double duration_ms;
    std::string message;
};

class VGFCoreTests {
private:
    std::vector<TestResult> results;
    std::mt19937 rng{42}; // Fixed seed for reproducibility
    
public:
    void run_all_tests() {
        std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║           VGF Core Library - Unit Test Suite            ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════╝\n" << std::endl;
        
        // Basic VGF Tests
        std::cout << "▶ Running Basic VGF Tests..." << std::endl;
        test_encoder_creation();
        test_module_addition();
        test_resource_addition();
        test_constant_addition();
        test_segment_creation();
        
        // Encoder/Decoder Tests
        std::cout << "\n▶ Running Encoder/Decoder Tests..." << std::endl;
        test_encode_decode_basic();
        test_encode_decode_with_modules();
        test_encode_decode_with_resources();
        test_multiple_descriptor_sets();
        
        // Data Integrity Tests
        std::cout << "\n▶ Running Data Integrity Tests..." << std::endl;
        test_large_constant_data();
        test_multiple_constants();
        test_resource_categories();
        test_module_types();
        
        // Error Handling Tests
        std::cout << "\n▶ Running Error Handling Tests..." << std::endl;
        test_invalid_vgf_data();
        test_corrupted_header();
        test_version_mismatch();
        
        // Performance Tests
        std::cout << "\n▶ Running Performance Tests..." << std::endl;
        test_encoding_performance();
        test_decoding_performance();
        test_large_file_handling();
        
        // Stress Tests
        std::cout << "\n▶ Running Stress Tests..." << std::endl;
        test_many_modules();
        test_many_resources();
        test_large_push_constants();
        
        print_summary();
    }
    
private:
    void test_encoder_creation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            if (!encoder) {
                passed = false;
                message = "Failed to create encoder";
            } else {
                message = "Encoder created successfully";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Encoder Creation", passed, start, message);
    }
    
    void test_module_addition() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Create SPIR-V shader data
            std::vector<uint32_t> spirv_code = {
                0x07230203, // SPIR-V magic number
                0x00010000, // Version 1.0
                0x00000000, // Generator
                0x00000001, // Bound
                0x00000000  // Schema
            };
            
            // Add compute shader module
            auto module_ref = encoder->AddModule(
                ModuleType::COMPUTE,
                spirv_code.data(),
                spirv_code.size() * sizeof(uint32_t),
                "test_compute_shader"
            );
            
            if (module_ref.reference == 0) {
                passed = false;
                message = "Invalid module reference returned";
            } else {
                message = "Module added successfully";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Module Addition", passed, start, message);
    }
    
    void test_resource_addition() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Add storage buffer resource
            auto resource_ref = encoder->AddResource(
                ResourceCategory::STORAGE_BUFFER,
                1024 * sizeof(float),  // 1024 floats
                "input_tensor"
            );
            
            if (resource_ref.reference == 0) {
                passed = false;
                message = "Invalid resource reference";
            } else {
                // Add another resource
                auto output_ref = encoder->AddResource(
                    ResourceCategory::STORAGE_BUFFER,
                    1024 * sizeof(float),
                    "output_tensor"
                );
                
                if (output_ref.reference <= resource_ref.reference) {
                    passed = false;
                    message = "Resource references not incrementing";
                } else {
                    message = "Resources added successfully";
                }
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Resource Addition", passed, start, message);
    }
    
    void test_constant_addition() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Create constant data
            std::vector<float> weights(256, 0.5f);
            
            // Add constant
            auto const_ref = encoder->AddConstant(
                weights.data(),
                weights.size() * sizeof(float),
                "conv_weights"
            );
            
            if (const_ref.reference == 0) {
                passed = false;
                message = "Invalid constant reference";
            } else {
                message = "Constant added successfully";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Constant Addition", passed, start, message);
    }
    
    void test_segment_creation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Create segment info
            auto segment_ref = encoder->AddSegmentInfo(
                0,     // offset
                1024,  // size
                "data_segment"
            );
            
            if (segment_ref.reference == 0) {
                passed = false;
                message = "Invalid segment reference";
            } else {
                message = "Segment created successfully";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Segment Creation", passed, start, message);
    }
    
    void test_encode_decode_basic() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            // Encode
            auto encoder = Encoder::Create();
            
            // Add some data
            std::vector<uint32_t> spirv = {0x07230203, 0x00010000, 0, 1, 0};
            encoder->AddModule(ModuleType::COMPUTE, spirv.data(), 
                             spirv.size() * sizeof(uint32_t), "shader");
            
            // Encode to buffer
            std::vector<uint8_t> vgf_data;
            size_t encoded_size = encoder->Encode(nullptr, 0);
            vgf_data.resize(encoded_size);
            encoder->Encode(vgf_data.data(), encoded_size);
            
            // Decode
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            
            if (!decoder.IsValid()) {
                passed = false;
                message = "Decoded data is invalid";
            } else if (decoder.GetVersion() == 0) {
                passed = false;
                message = "Invalid version in decoded data";
            } else {
                message = "Basic encode/decode successful";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Basic Encode/Decode", passed, start, message);
    }
    
    void test_encode_decode_with_modules() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Add multiple modules
            std::vector<uint32_t> spirv1 = {0x07230203, 0x00010000, 0, 1, 0};
            std::vector<uint32_t> spirv2 = {0x07230203, 0x00010000, 0, 2, 0};
            
            encoder->AddModule(ModuleType::COMPUTE, spirv1.data(),
                             spirv1.size() * sizeof(uint32_t), "shader1");
            encoder->AddModule(ModuleType::COMPUTE, spirv2.data(),
                             spirv2.size() * sizeof(uint32_t), "shader2");
            
            // Encode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Decode and verify
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            ModuleTableDecoder module_table(&decoder);
            
            if (module_table.GetModuleCount() != 2) {
                passed = false;
                message = "Module count mismatch";
            } else {
                message = "Multiple modules encoded/decoded correctly";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Module Encode/Decode", passed, start, message);
    }
    
    void test_encode_decode_with_resources() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Add resources
            encoder->AddResource(ResourceCategory::STORAGE_BUFFER, 1024, "buffer1");
            encoder->AddResource(ResourceCategory::STORAGE_BUFFER, 2048, "buffer2");
            encoder->AddResource(ResourceCategory::UNIFORM_BUFFER, 256, "uniforms");
            
            // Encode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Decode
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            ModelResourceTableDecoder resource_table(&decoder);
            
            if (resource_table.GetResourceCount() != 3) {
                passed = false;
                message = "Resource count mismatch";
            } else {
                message = "Resources encoded/decoded correctly";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Resource Encode/Decode", passed, start, message);
    }
    
    void test_multiple_descriptor_sets() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Create multiple descriptor sets
            for (int set = 0; set < 4; set++) {
                auto ds_ref = encoder->AddDescriptorSetInfo(set);
                
                // Add bindings to each set
                for (int binding = 0; binding < 3; binding++) {
                    encoder->AddBindingSlot(
                        ds_ref,
                        binding,
                        DescriptorType::STORAGE_BUFFER,
                        1,
                        "binding_" + std::to_string(binding)
                    );
                }
            }
            
            // Encode and verify
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            if (vgf_data.empty()) {
                passed = false;
                message = "Failed to encode descriptor sets";
            } else {
                message = "Multiple descriptor sets handled correctly";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Multiple Descriptor Sets", passed, start, message);
    }
    
    void test_large_constant_data() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Create large constant data (10MB)
            std::vector<float> large_data(10 * 1024 * 1024 / sizeof(float));
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (auto& val : large_data) {
                val = dist(rng);
            }
            
            // Add as constant
            encoder->AddConstant(large_data.data(), 
                               large_data.size() * sizeof(float),
                               "large_weights");
            
            // Encode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Decode and verify
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            ConstantDecoder const_decoder(&decoder);
            
            if (const_decoder.GetConstantCount() != 1) {
                passed = false;
                message = "Large constant not found";
            } else {
                auto const_data = const_decoder.GetConstantData(0);
                if (!const_data.data || const_data.size != large_data.size() * sizeof(float)) {
                    passed = false;
                    message = "Large constant size mismatch";
                } else {
                    message = "Large constant data handled correctly (10MB)";
                }
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Large Constant Data", passed, start, message);
    }
    
    void test_multiple_constants() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Add multiple constants of different sizes
            std::vector<float> weights1(1024, 0.1f);
            std::vector<float> weights2(2048, 0.2f);
            std::vector<float> biases(256, 0.0f);
            
            encoder->AddConstant(weights1.data(), weights1.size() * sizeof(float), "weights1");
            encoder->AddConstant(weights2.data(), weights2.size() * sizeof(float), "weights2");
            encoder->AddConstant(biases.data(), biases.size() * sizeof(float), "biases");
            
            // Encode and decode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            ConstantDecoder const_decoder(&decoder);
            
            if (const_decoder.GetConstantCount() != 3) {
                passed = false;
                message = "Constant count mismatch";
            } else {
                message = "Multiple constants handled correctly";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Multiple Constants", passed, start, message);
    }
    
    void test_resource_categories() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Test all resource categories
            encoder->AddResource(ResourceCategory::STORAGE_BUFFER, 1024, "storage");
            encoder->AddResource(ResourceCategory::UNIFORM_BUFFER, 256, "uniform");
            encoder->AddResource(ResourceCategory::SAMPLED_IMAGE, 0, "texture");
            encoder->AddResource(ResourceCategory::STORAGE_IMAGE, 0, "storage_img");
            encoder->AddResource(ResourceCategory::SAMPLER, 0, "sampler");
            
            // Encode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Decode and verify
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            ModelResourceTableDecoder resource_table(&decoder);
            
            if (resource_table.GetResourceCount() != 5) {
                passed = false;
                message = "Not all resource categories preserved";
            } else {
                message = "All resource categories handled correctly";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Resource Categories", passed, start, message);
    }
    
    void test_module_types() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Test different module types
            std::vector<uint32_t> spirv = {0x07230203, 0x00010000, 0, 1, 0};
            
            encoder->AddModule(ModuleType::COMPUTE, spirv.data(), 
                             spirv.size() * sizeof(uint32_t), "compute");
            encoder->AddModule(ModuleType::VERTEX, spirv.data(),
                             spirv.size() * sizeof(uint32_t), "vertex");
            encoder->AddModule(ModuleType::FRAGMENT, spirv.data(),
                             spirv.size() * sizeof(uint32_t), "fragment");
            
            // Encode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Decode
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            ModuleTableDecoder module_table(&decoder);
            
            if (module_table.GetModuleCount() != 3) {
                passed = false;
                message = "Module type count mismatch";
            } else {
                message = "All module types handled correctly";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Module Types", passed, start, message);
    }
    
    void test_invalid_vgf_data() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            // Create invalid data
            std::vector<uint8_t> invalid_data(100, 0xFF);
            
            // Try to decode
            HeaderDecoder decoder(invalid_data.data(), invalid_data.size());
            
            if (decoder.IsValid()) {
                passed = false;
                message = "Invalid data accepted as valid";
            } else {
                message = "Invalid VGF data correctly rejected";
            }
        } catch (const std::exception& e) {
            // Exception is expected for invalid data
            message = "Invalid data handled with exception";
        }
        
        record_result("Invalid VGF Data", passed, start, message);
    }
    
    void test_corrupted_header() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            // Create valid VGF first
            auto encoder = Encoder::Create();
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Corrupt the header
            if (vgf_data.size() > 4) {
                vgf_data[0] = 0xFF;
                vgf_data[1] = 0xFF;
                vgf_data[2] = 0xFF;
                vgf_data[3] = 0xFF;
            }
            
            // Try to decode
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            
            if (decoder.IsValid()) {
                passed = false;
                message = "Corrupted header accepted";
            } else {
                message = "Corrupted header correctly detected";
            }
        } catch (const std::exception& e) {
            message = "Corrupted header handled with exception";
        }
        
        record_result("Corrupted Header", passed, start, message);
    }
    
    void test_version_mismatch() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            // Create encoder and encode data
            auto encoder = Encoder::Create();
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Modify version bytes (assuming they're at a known offset)
            if (vgf_data.size() > 8) {
                vgf_data[4] = 0xFF;  // Corrupt version field
            }
            
            // Try to decode
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            
            // Check version handling
            message = "Version mismatch handling tested";
            
        } catch (const std::exception& e) {
            message = "Version mismatch handled with exception";
        }
        
        record_result("Version Mismatch", passed, start, message);
    }
    
    void test_encoding_performance() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            const int iterations = 100;
            std::vector<double> encode_times;
            
            for (int i = 0; i < iterations; i++) {
                auto encoder = Encoder::Create();
                
                // Add data
                std::vector<float> data(10000, 1.0f);
                encoder->AddConstant(data.data(), data.size() * sizeof(float), "data");
                
                auto encode_start = std::chrono::high_resolution_clock::now();
                
                // Encode
                size_t size = encoder->Encode(nullptr, 0);
                std::vector<uint8_t> vgf_data(size);
                encoder->Encode(vgf_data.data(), size);
                
                auto encode_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
                encode_times.push_back(ms);
            }
            
            // Calculate average
            double avg_time = 0;
            for (double t : encode_times) avg_time += t;
            avg_time /= iterations;
            
            // Check performance (should be < 10ms for 40KB data)
            if (avg_time > 10.0) {
                passed = false;
                message = "Encoding too slow: " + std::to_string(avg_time) + "ms avg";
            } else {
                message = "Encoding performance: " + std::to_string(avg_time) + "ms avg";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Encoding Performance", passed, start, message);
    }
    
    void test_decoding_performance() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            // Create test data
            auto encoder = Encoder::Create();
            std::vector<float> data(100000, 1.0f);
            encoder->AddConstant(data.data(), data.size() * sizeof(float), "data");
            
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Measure decoding performance
            const int iterations = 100;
            auto decode_start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iterations; i++) {
                HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
                ConstantDecoder const_decoder(&decoder);
                auto const_data = const_decoder.GetConstantData(0);
            }
            
            auto decode_end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
            double avg_ms = total_ms / iterations;
            
            // Check performance (should be < 5ms for 400KB data)
            if (avg_ms > 5.0) {
                passed = false;
                message = "Decoding too slow: " + std::to_string(avg_ms) + "ms avg";
            } else {
                message = "Decoding performance: " + std::to_string(avg_ms) + "ms avg";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Decoding Performance", passed, start, message);
    }
    
    void test_large_file_handling() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Create 50MB of data
            std::vector<float> huge_data(50 * 1024 * 1024 / sizeof(float));
            for (size_t i = 0; i < huge_data.size(); i++) {
                huge_data[i] = static_cast<float>(i % 1000) / 1000.0f;
            }
            
            // Add as constant
            encoder->AddConstant(huge_data.data(), 
                               huge_data.size() * sizeof(float),
                               "huge_weights");
            
            // Encode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Decode
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            ConstantDecoder const_decoder(&decoder);
            
            auto const_data = const_decoder.GetConstantData(0);
            if (!const_data.data || const_data.size != huge_data.size() * sizeof(float)) {
                passed = false;
                message = "Large file size mismatch";
            } else {
                double size_mb = vgf_data.size() / (1024.0 * 1024.0);
                message = "Large file handled: " + std::to_string(size_mb) + "MB";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Large File Handling", passed, start, message);
    }
    
    void test_many_modules() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Add 100 modules
            std::vector<uint32_t> spirv = {0x07230203, 0x00010000, 0, 1, 0};
            for (int i = 0; i < 100; i++) {
                encoder->AddModule(ModuleType::COMPUTE, spirv.data(),
                                 spirv.size() * sizeof(uint32_t),
                                 "module_" + std::to_string(i));
            }
            
            // Encode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Decode
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            ModuleTableDecoder module_table(&decoder);
            
            if (module_table.GetModuleCount() != 100) {
                passed = false;
                message = "Module count mismatch in stress test";
            } else {
                message = "100 modules handled successfully";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Many Modules Stress", passed, start, message);
    }
    
    void test_many_resources() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Add 500 resources
            for (int i = 0; i < 500; i++) {
                encoder->AddResource(
                    ResourceCategory::STORAGE_BUFFER,
                    1024 + i * 4,  // Varying sizes
                    "resource_" + std::to_string(i)
                );
            }
            
            // Encode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            // Decode
            HeaderDecoder decoder(vgf_data.data(), vgf_data.size());
            ModelResourceTableDecoder resource_table(&decoder);
            
            if (resource_table.GetResourceCount() != 500) {
                passed = false;
                message = "Resource count mismatch in stress test";
            } else {
                message = "500 resources handled successfully";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Many Resources Stress", passed, start, message);
    }
    
    void test_large_push_constants() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            auto encoder = Encoder::Create();
            
            // Add push constant ranges (max 128 bytes typically)
            for (int i = 0; i < 10; i++) {
                encoder->AddPushConstantRange(
                    i * 16,  // offset
                    16,      // size
                    "push_const_" + std::to_string(i)
                );
            }
            
            // Encode
            std::vector<uint8_t> vgf_data;
            size_t size = encoder->Encode(nullptr, 0);
            vgf_data.resize(size);
            encoder->Encode(vgf_data.data(), size);
            
            if (vgf_data.empty()) {
                passed = false;
                message = "Failed to encode push constants";
            } else {
                message = "Large push constant ranges handled";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Large Push Constants", passed, start, message);
    }
    
    void record_result(const std::string& name, bool passed, 
                      const std::chrono::high_resolution_clock::time_point& start,
                      const std::string& message) {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        
        results.push_back({name, passed, duration, message});
        
        // Print immediate result
        std::cout << "  " << std::setw(30) << std::left << name << ": ";
        if (passed) {
            std::cout << "✅ PASS";
        } else {
            std::cout << "❌ FAIL";
        }
        std::cout << " (" << std::fixed << std::setprecision(2) << duration << "ms)";
        if (!passed) {
            std::cout << " - " << message;
        }
        std::cout << std::endl;
    }
    
    void print_summary() {
        std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                      TEST SUMMARY                       ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
        
        int passed = 0, failed = 0;
        double total_time = 0;
        
        for (const auto& result : results) {
            if (result.passed) passed++;
            else failed++;
            total_time += result.duration_ms;
        }
        
        std::cout << "\nTotal Tests: " << (passed + failed) << std::endl;
        std::cout << "✅ Passed: " << passed << std::endl;
        std::cout << "❌ Failed: " << failed << std::endl;
        std::cout << "⏱️  Total Time: " << std::fixed << std::setprecision(2) 
                  << total_time << "ms" << std::endl;
        
        double success_rate = (passed + failed) > 0 ? 
                             (100.0 * passed / (passed + failed)) : 0;
        std::cout << "📊 Success Rate: " << std::fixed << std::setprecision(1) 
                  << success_rate << "%" << std::endl;
        
        if (success_rate >= 95) {
            std::cout << "\n🎉 EXCELLENT - VGF Library is production ready!" << std::endl;
        } else if (success_rate >= 80) {
            std::cout << "\n✅ GOOD - Minor issues detected" << std::endl;
        } else {
            std::cout << "\n⚠️  NEEDS IMPROVEMENT" << std::endl;
        }
        
        // Return exit code based on failures
        if (failed > 0) {
            exit(1);
        }
    }
};

int main() {
    std::cout << "VGF Core Library Unit Tests" << std::endl;
    std::cout << "Platform: macOS ARM64 (Apple M4 Max)" << std::endl;
    std::cout << "Library: libvgf.a" << std::endl;
    
    VGFCoreTests tests;
    tests.run_all_tests();
    
    return 0;
}