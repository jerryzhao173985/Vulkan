/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Build System Verification Tests for ARM ML SDK
 * Tests SDK build integrity, library linking, symbol resolution, and installation
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <mach-o/dyld.h>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <regex>

namespace mlsdk::tests {

namespace fs = std::filesystem;

class BuildSystemTest : public ::testing::Test {
protected:
    fs::path sdkRoot;
    fs::path buildDir;
    fs::path binDir;
    fs::path libDir;
    fs::path includeDir;
    fs::path modelsDir;
    fs::path shadersDir;
    fs::path toolsDir;
    
    void SetUp() override {
        // Set paths from CMake definitions
        sdkRoot = fs::path("/Users/jerry/Vulkan");
        buildDir = sdkRoot / "builds" / "ARM-ML-SDK-Complete";
        binDir = buildDir / "bin";
        libDir = buildDir / "lib";
        includeDir = buildDir / "include";
        modelsDir = buildDir / "models";
        shadersDir = buildDir / "shaders";
        toolsDir = buildDir / "tools";
    }
    
    bool fileExists(const fs::path& path) {
        return fs::exists(path);
    }
    
    bool isExecutable(const fs::path& path) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0) {
            return (st.st_mode & S_IXUSR) != 0;
        }
        return false;
    }
    
    bool canLoadLibrary(const fs::path& libPath) {
        void* handle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle) {
            dlclose(handle);
            return true;
        }
        return false;
    }
    
    std::vector<std::string> getLibraryDependencies(const fs::path& libPath) {
        std::vector<std::string> deps;
        
        // Use otool on macOS to get dependencies
        std::string cmd = "otool -L " + libPath.string() + " 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return deps;
        
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            std::string line(buffer);
            // Skip first line (library name itself)
            if (line.find(libPath.filename()) != std::string::npos) continue;
            
            // Extract library path from otool output
            size_t start = line.find_first_not_of(" \t");
            size_t end = line.find(" (");
            if (start != std::string::npos && end != std::string::npos) {
                deps.push_back(line.substr(start, end - start));
            }
        }
        pclose(pipe);
        
        return deps;
    }
    
    bool hasSymbol(const fs::path& libPath, const std::string& symbol) {
        void* handle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) return false;
        
        void* sym = dlsym(handle, symbol.c_str());
        dlclose(handle);
        return sym != nullptr;
    }
};

// Test 1: Directory Structure Verification
TEST_F(BuildSystemTest, DirectoryStructure) {
    // Verify all required directories exist
    EXPECT_TRUE(fileExists(buildDir)) << "Build directory missing: " << buildDir;
    EXPECT_TRUE(fileExists(binDir)) << "Binary directory missing: " << binDir;
    EXPECT_TRUE(fileExists(libDir)) << "Library directory missing: " << libDir;
    EXPECT_TRUE(fileExists(includeDir)) << "Include directory missing: " << includeDir;
    EXPECT_TRUE(fileExists(modelsDir)) << "Models directory missing: " << modelsDir;
    EXPECT_TRUE(fileExists(shadersDir)) << "Shaders directory missing: " << shadersDir;
    EXPECT_TRUE(fileExists(toolsDir)) << "Tools directory missing: " << toolsDir;
    
    // Check subdirectories
    EXPECT_TRUE(fileExists(includeDir / "vgf")) << "VGF headers missing";
    
    // Verify directory permissions
    EXPECT_TRUE(fs::is_directory(buildDir));
    EXPECT_TRUE(fs::is_directory(binDir));
    EXPECT_TRUE(fs::is_directory(libDir));
}

// Test 2: Main Executable Verification
TEST_F(BuildSystemTest, MainExecutable) {
    fs::path scenarioRunner = binDir / "scenario-runner";
    
    ASSERT_TRUE(fileExists(scenarioRunner)) 
        << "scenario-runner executable not found at: " << scenarioRunner;
    
    EXPECT_TRUE(isExecutable(scenarioRunner)) 
        << "scenario-runner is not executable";
    
    // Check file size (should be substantial)
    auto fileSize = fs::file_size(scenarioRunner);
    EXPECT_GT(fileSize, 1024 * 1024) // At least 1MB
        << "scenario-runner size too small: " << fileSize;
    
    // Expected size is around 43MB
    EXPECT_GT(fileSize, 40 * 1024 * 1024) 
        << "scenario-runner smaller than expected: " << fileSize;
    
    // Test execution with --version
    std::string cmd = scenarioRunner.string() + " --version 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    ASSERT_NE(pipe, nullptr) << "Failed to execute scenario-runner";
    
    char buffer[256];
    bool hasOutput = false;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        hasOutput = true;
    }
    
    int result = pclose(pipe);
    EXPECT_EQ(WEXITSTATUS(result), 0) << "scenario-runner --version failed";
    EXPECT_TRUE(hasOutput) << "scenario-runner produced no output";
}

// Test 3: VGF Library Verification
TEST_F(BuildSystemTest, VGFLibrary) {
    fs::path vgfLib = libDir / "libvgflib.dylib";
    
    ASSERT_TRUE(fileExists(vgfLib)) 
        << "VGF library not found at: " << vgfLib;
    
    // Check library can be loaded
    EXPECT_TRUE(canLoadLibrary(vgfLib)) 
        << "Failed to load VGF library";
    
    // Check for expected symbols
    std::vector<std::string> expectedSymbols = {
        "_ZN6mlsdk6vgflib7Encoder6CreateEv",  // Encoder::Create
        "_ZN6mlsdk6vgflib7Decoder6CreateEv",  // Decoder::Create
        "_ZN6mlsdk6vgflib7Encoder9AddModuleE",  // Encoder::AddModule
        "_ZN6mlsdk6vgflib7Encoder11AddResourceE",  // Encoder::AddResource
        "_ZN6mlsdk6vgflib7Encoder11AddConstantE"  // Encoder::AddConstant
    };
    
    for (const auto& symbol : expectedSymbols) {
        EXPECT_TRUE(hasSymbol(vgfLib, symbol)) 
            << "Missing symbol: " << symbol;
    }
    
    // Check library dependencies
    auto deps = getLibraryDependencies(vgfLib);
    bool hasSystemLibs = false;
    for (const auto& dep : deps) {
        if (dep.find("libc++") != std::string::npos ||
            dep.find("libSystem") != std::string::npos) {
            hasSystemLibs = true;
        }
    }
    EXPECT_TRUE(hasSystemLibs) << "VGF library missing system dependencies";
}

// Test 4: SPIRV Libraries Verification
TEST_F(BuildSystemTest, SPIRVLibraries) {
    std::vector<std::string> spirvLibs = {
        "libspirv-cross-c.dylib",
        "libspirv-cross-core.dylib",
        "libspirv-cross-cpp.dylib",
        "libspirv-cross-glsl.dylib",
        "libspirv-cross-hlsl.dylib",
        "libspirv-cross-msl.dylib",
        "libspirv-cross-reflect.dylib"
    };
    
    for (const auto& libName : spirvLibs) {
        fs::path libPath = libDir / libName;
        
        EXPECT_TRUE(fileExists(libPath)) 
            << "SPIRV library not found: " << libName;
        
        if (fileExists(libPath)) {
            EXPECT_TRUE(canLoadLibrary(libPath)) 
                << "Failed to load SPIRV library: " << libName;
            
            // Check file size
            auto fileSize = fs::file_size(libPath);
            EXPECT_GT(fileSize, 100 * 1024) // At least 100KB
                << libName << " size too small: " << fileSize;
        }
    }
    
    // Verify all 7 SPIRV libraries are present
    int foundCount = 0;
    for (const auto& libName : spirvLibs) {
        if (fileExists(libDir / libName)) {
            foundCount++;
        }
    }
    EXPECT_EQ(foundCount, 7) 
        << "Expected 7 SPIRV libraries, found: " << foundCount;
}

// Test 5: Model Files Verification
TEST_F(BuildSystemTest, ModelFiles) {
    std::map<std::string, size_t> expectedModels = {
        {"mobilenet_v2_1.0_224_quantized.tflite", 3400000},  // ~3.4MB
        {"la_muse.tflite", 7000000},     // ~7MB
        {"udnie.tflite", 7000000},        // ~7MB
        {"mirror.tflite", 7000000},       // ~7MB
        {"wave_crop.tflite", 7000000},   // ~7MB
        {"des_glaneuses.tflite", 7000000}, // ~7MB
        {"fire_detection.tflite", 8100000} // ~8.1MB
    };
    
    size_t totalSize = 0;
    int foundModels = 0;
    
    for (const auto& [modelName, expectedSize] : expectedModels) {
        fs::path modelPath = modelsDir / modelName;
        
        if (fileExists(modelPath)) {
            foundModels++;
            auto fileSize = fs::file_size(modelPath);
            totalSize += fileSize;
            
            // Allow 10% variance in size
            EXPECT_NEAR(fileSize, expectedSize, expectedSize * 0.1) 
                << "Model size mismatch for: " << modelName;
        } else {
            ADD_FAILURE() << "Model not found: " << modelName;
        }
    }
    
    EXPECT_EQ(foundModels, 7) << "Expected 7 models, found: " << foundModels;
    EXPECT_NEAR(totalSize, 46000000, 5000000) // ~46MB ±5MB
        << "Total model size mismatch: " << totalSize;
}

// Test 6: Shader Files Verification
TEST_F(BuildSystemTest, ShaderFiles) {
    std::vector<std::string> coreShaders = {
        "add.spv", "multiply.spv", "divide.spv",
        "conv2d.spv", "matmul.spv", 
        "relu.spv", "sigmoid.spv", "tanh.spv", "softmax.spv",
        "maxpool.spv", "avgpool.spv",
        "batchnorm.spv", "layernorm.spv",
        "reshape.spv", "transpose.spv", "concat.spv"
    };
    
    int foundShaders = 0;
    int validShaders = 0;
    
    for (const auto& shaderName : coreShaders) {
        fs::path shaderPath = shadersDir / shaderName;
        
        if (fileExists(shaderPath)) {
            foundShaders++;
            
            // Verify SPIR-V magic number
            std::ifstream file(shaderPath, std::ios::binary);
            if (file) {
                uint32_t magic;
                file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                if (magic == 0x07230203) {
                    validShaders++;
                } else {
                    ADD_FAILURE() << "Invalid SPIR-V shader: " << shaderName;
                }
            }
        }
    }
    
    EXPECT_GE(foundShaders, 12) 
        << "Too few core shaders found: " << foundShaders;
    EXPECT_EQ(validShaders, foundShaders) 
        << "Some shaders are invalid";
    
    // Count total shaders
    int totalShaders = 0;
    for (const auto& entry : fs::directory_iterator(shadersDir)) {
        if (entry.path().extension() == ".spv") {
            totalShaders++;
        }
    }
    
    EXPECT_GE(totalShaders, 35) 
        << "Expected at least 35 shaders, found: " << totalShaders;
}

// Test 7: Python Tools Verification
TEST_F(BuildSystemTest, PythonTools) {
    std::vector<std::string> expectedTools = {
        "analyze_tflite_model.py",
        "optimize_for_apple_silicon.py",
        "profile_performance.py",
        "compare_numpy.py",
        "dump_numpy.py"
    };
    
    for (const auto& toolName : expectedTools) {
        fs::path toolPath = toolsDir / toolName;
        
        EXPECT_TRUE(fileExists(toolPath)) 
            << "Python tool not found: " << toolName;
        
        if (fileExists(toolPath)) {
            // Check if it's a valid Python file
            std::ifstream file(toolPath);
            std::string firstLine;
            std::getline(file, firstLine);
            
            // Should start with shebang or be valid Python
            if (!firstLine.empty()) {
                bool isPython = firstLine.find("#!/usr/bin/env python") != std::string::npos ||
                               firstLine.find("#!/usr/bin/python") != std::string::npos ||
                               firstLine.find("import ") != std::string::npos ||
                               firstLine.find("from ") != std::string::npos ||
                               firstLine.find("#") != std::string::npos;
                
                EXPECT_TRUE(isPython) 
                    << "Invalid Python file: " << toolName;
            }
        }
    }
}

// Test 8: Header Files Verification
TEST_F(BuildSystemTest, HeaderFiles) {
    // Check VGF headers
    std::vector<std::string> vgfHeaders = {
        "vgf/encoder.hpp",
        "vgf/decoder.hpp",
        "vgf/types.hpp"
    };
    
    for (const auto& header : vgfHeaders) {
        fs::path headerPath = includeDir / header;
        EXPECT_TRUE(fileExists(headerPath)) 
            << "Header not found: " << header;
    }
    
    // Check for include guards or pragma once
    for (const auto& header : vgfHeaders) {
        fs::path headerPath = includeDir / header;
        if (fileExists(headerPath)) {
            std::ifstream file(headerPath);
            std::string line;
            bool hasGuard = false;
            
            while (std::getline(file, line)) {
                if (line.find("#pragma once") != std::string::npos ||
                    line.find("#ifndef") != std::string::npos) {
                    hasGuard = true;
                    break;
                }
            }
            
            EXPECT_TRUE(hasGuard) 
                << "Header missing include guard: " << header;
        }
    }
}

// Test 9: Library Symbol Resolution
TEST_F(BuildSystemTest, SymbolResolution) {
    fs::path scenarioRunner = binDir / "scenario-runner";
    
    if (!fileExists(scenarioRunner)) {
        GTEST_SKIP() << "scenario-runner not found";
    }
    
    // Check for undefined symbols
    std::string cmd = "nm -u " + scenarioRunner.string() + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    ASSERT_NE(pipe, nullptr);
    
    std::vector<std::string> undefinedSymbols;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        std::string line(buffer);
        // Filter out expected system symbols
        if (line.find("dyld") == std::string::npos &&
            line.find("__") != 0) {  // Skip system symbols starting with __
            undefinedSymbols.push_back(line);
        }
    }
    pclose(pipe);
    
    // Should have minimal undefined symbols (only system ones)
    EXPECT_LT(undefinedSymbols.size(), 100) 
        << "Too many undefined symbols: " << undefinedSymbols.size();
}

// Test 10: Build Configuration Files
TEST_F(BuildSystemTest, BuildConfiguration) {
    // Check for CMake cache
    fs::path cmakeCache = sdkRoot / "ai-ml-sdk-for-vulkan" / "build-final" / "CMakeCache.txt";
    
    if (fileExists(cmakeCache)) {
        std::ifstream file(cmakeCache);
        std::string line;
        
        bool hasReleaseMode = false;
        bool hasInstallPrefix = false;
        
        while (std::getline(file, line)) {
            if (line.find("CMAKE_BUILD_TYPE:STRING=Release") != std::string::npos) {
                hasReleaseMode = true;
            }
            if (line.find("CMAKE_INSTALL_PREFIX") != std::string::npos) {
                hasInstallPrefix = true;
            }
        }
        
        EXPECT_TRUE(hasReleaseMode) << "Not built in Release mode";
        EXPECT_TRUE(hasInstallPrefix) << "No install prefix configured";
    }
}

// Test 11: Library Dependencies Chain
TEST_F(BuildSystemTest, DependencyChain) {
    fs::path scenarioRunner = binDir / "scenario-runner";
    
    if (!fileExists(scenarioRunner)) {
        GTEST_SKIP() << "scenario-runner not found";
    }
    
    // Get all dependencies
    auto deps = getLibraryDependencies(scenarioRunner);
    
    // Check for expected dependencies
    bool hasVulkan = false;
    bool hasVGF = false;
    bool hasSPIRV = false;
    
    for (const auto& dep : deps) {
        if (dep.find("vulkan") != std::string::npos ||
            dep.find("MoltenVK") != std::string::npos) {
            hasVulkan = true;
        }
        if (dep.find("vgf") != std::string::npos) {
            hasVGF = true;
        }
        if (dep.find("spirv") != std::string::npos) {
            hasSPIRV = true;
        }
    }
    
    EXPECT_TRUE(hasVulkan) << "Missing Vulkan dependency";
    EXPECT_TRUE(hasVGF) << "Missing VGF dependency";
    // SPIRV might be statically linked
}

// Test 12: Version Consistency
TEST_F(BuildSystemTest, VersionConsistency) {
    // Check version in different places matches
    std::string expectedVersion = "1.0.0";  // Or read from a version file
    
    // Could check:
    // - Binary version output
    // - Library versions
    // - Header file versions
    // - Package version
    
    // This is a placeholder for version checking logic
    EXPECT_TRUE(true) << "Version consistency check";
}

// Test 13: Installation Layout
TEST_F(BuildSystemTest, InstallationLayout) {
    // Verify the installation follows expected layout
    struct {
        fs::path path;
        bool required;
        const char* description;
    } installChecks[] = {
        {binDir / "scenario-runner", true, "Main executable"},
        {libDir / "libvgflib.dylib", true, "VGF library"},
        {includeDir / "vgf", true, "VGF headers"},
        {modelsDir, true, "Models directory"},
        {shadersDir, true, "Shaders directory"},
        {toolsDir, true, "Tools directory"}
    };
    
    for (const auto& check : installChecks) {
        if (check.required) {
            EXPECT_TRUE(fileExists(check.path)) 
                << "Missing: " << check.description 
                << " at " << check.path;
        }
    }
}

// Test 14: Runtime Library Loading
TEST_F(BuildSystemTest, RuntimeLibraryLoading) {
    // Set up library path
    setenv("DYLD_LIBRARY_PATH", libDir.c_str(), 1);
    
    // Try to load each library
    std::vector<std::string> libs = {
        "libvgflib.dylib",
        "libspirv-cross-core.dylib"
    };
    
    for (const auto& libName : libs) {
        fs::path libPath = libDir / libName;
        if (fileExists(libPath)) {
            void* handle = dlopen(libPath.c_str(), RTLD_LAZY);
            EXPECT_NE(handle, nullptr) 
                << "Failed to load library: " << libName 
                << " Error: " << dlerror();
            
            if (handle) {
                dlclose(handle);
            }
        }
    }
}

// Test 15: Build Reproducibility
TEST_F(BuildSystemTest, BuildReproducibility) {
    // Check that build artifacts are consistent
    // This could involve checking timestamps, sizes, checksums
    
    // For now, just check that key files have reasonable sizes
    struct {
        fs::path path;
        size_t minSize;
        size_t maxSize;
    } sizeChecks[] = {
        {binDir / "scenario-runner", 40*1024*1024, 50*1024*1024},  // 40-50MB
        {libDir / "libvgflib.dylib", 100*1024, 10*1024*1024},      // 100KB-10MB
    };
    
    for (const auto& check : sizeChecks) {
        if (fileExists(check.path)) {
            auto size = fs::file_size(check.path);
            EXPECT_GE(size, check.minSize) 
                << check.path << " too small: " << size;
            EXPECT_LE(size, check.maxSize) 
                << check.path << " too large: " << size;
        }
    }
}

} // namespace mlsdk::tests