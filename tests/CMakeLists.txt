# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Comprehensive Test Suite CMake Configuration
# ARM ML SDK for Vulkan - All Test Components

cmake_minimum_required(VERSION 3.25)
project(VulkanMLSDKTests VERSION 1.0.0 LANGUAGES CXX C)

# C++ Standard Requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform detection
if(APPLE)
    set(PLATFORM_MACOS TRUE)
    # Apple Silicon optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(APPLE_SILICON TRUE)
        add_compile_options(-mcpu=apple-m1)
    endif()
endif()

# Options
option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_INTEGRATION_TESTS "Build integration tests" ON)
option(BUILD_PERFORMANCE_TESTS "Build performance tests" ON)
option(BUILD_STRESS_TESTS "Build stress tests" ON)
option(BUILD_PLATFORM_TESTS "Build platform-specific tests" ON)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_SANITIZERS "Enable address and thread sanitizers" OFF)
option(PARALLEL_TEST_EXECUTION "Enable parallel test execution" ON)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Python for test scripts
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Set SDK paths
set(SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(SDK_BUILD_DIR "${SDK_ROOT}/builds/ARM-ML-SDK-Complete")
set(SDK_BIN_DIR "${SDK_BUILD_DIR}/bin")
set(SDK_LIB_DIR "${SDK_BUILD_DIR}/lib")
set(SDK_INCLUDE_DIR "${SDK_BUILD_DIR}/include")
set(SDK_MODELS_DIR "${SDK_BUILD_DIR}/models")
set(SDK_SHADERS_DIR "${SDK_BUILD_DIR}/shaders")

# VGF Library paths
set(VGF_INCLUDE_DIR "${SDK_ROOT}/ai-ml-sdk-vgf-library/include")
set(VGF_LIB_DIR "${SDK_LIB_DIR}")

# Include directories
include_directories(
    ${Vulkan_INCLUDE_DIRS}
    ${SDK_INCLUDE_DIR}
    ${VGF_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/framework
)

# Link directories
link_directories(
    ${SDK_LIB_DIR}
    ${VGF_LIB_DIR}
)

# Common test libraries
set(COMMON_TEST_LIBS
    ${Vulkan_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    vgflib
)

# SPIRV libraries
set(SPIRV_LIBS
    spirv-cross-c
    spirv-cross-core
    spirv-cross-cpp
    spirv-cross-glsl
    spirv-cross-hlsl
    spirv-cross-msl
    spirv-cross-reflect
)

# Platform-specific libraries
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    list(APPEND COMMON_TEST_LIBS 
        ${METAL_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
    )
    
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        add_compile_options(-g -O0)
    elseif(CMAKE_BUILD_TYPE MATCHES Release)
        add_compile_options(-O3 -march=native)
    endif()
    
    if(ENABLE_SANITIZERS)
        add_compile_options(-fsanitize=address -fsanitize=undefined)
        add_link_options(-fsanitize=address -fsanitize=undefined)
    endif()
    
    if(ENABLE_COVERAGE)
        add_compile_options(--coverage)
        add_link_options(--coverage)
    endif()
endif()

# Test data directory
set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
add_compile_definitions(TEST_DATA_DIR="${TEST_DATA_DIR}")
add_compile_definitions(SDK_BIN_PATH="${SDK_BIN_DIR}")
add_compile_definitions(SDK_MODELS_PATH="${SDK_MODELS_DIR}")
add_compile_definitions(SDK_SHADERS_PATH="${SDK_SHADERS_DIR}")

# === UNIT TESTS ===
if(BUILD_UNIT_TESTS)
    # Vulkan Core Tests
    add_executable(vulkan_core_tests
        unit/vulkan_core_tests.cpp
    )
    target_link_libraries(vulkan_core_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # SPIRV Shader Tests
    add_executable(spirv_shader_tests
        unit/spirv_shader_tests.cpp
    )
    target_link_libraries(spirv_shader_tests PRIVATE ${COMMON_TEST_LIBS} ${SPIRV_LIBS})
    
    # Memory Management Tests
    add_executable(memory_management_tests
        unit/memory_management_tests.cpp
    )
    target_link_libraries(memory_management_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # ML Operations Tests
    add_executable(ml_operations_tests
        unit/ml_operations_tests.cpp
    )
    target_link_libraries(ml_operations_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # VGF Core Tests
    add_executable(vgf_core_tests
        unit/vgf/test_vgf_core.cpp
    )
    target_link_libraries(vgf_core_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # VGF Simple Tests
    add_executable(vgf_simple_tests
        unit/vgf/test_vgf_simple.cpp
    )
    target_link_libraries(vgf_simple_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # VGF Minimal Tests
    add_executable(vgf_minimal_tests
        unit/vgf/test_vgf_minimal.cpp
    )
    target_link_libraries(vgf_minimal_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # Scenario Runner Tests
    add_executable(scenario_runner_tests
        unit/runner/test_scenario_runner.cpp
    )
    target_link_libraries(scenario_runner_tests PRIVATE ${COMMON_TEST_LIBS})
endif()

# === INTEGRATION TESTS ===
if(BUILD_INTEGRATION_TESTS)
    # Model Pipeline Tests
    add_executable(model_pipeline_tests
        integration/model_pipeline/test_model_pipeline.cpp
    )
    target_link_libraries(model_pipeline_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # Scenario Execution Tests
    add_executable(scenario_execution_tests
        integration/scenario_execution/test_scenario_execution.cpp
    )
    target_link_libraries(scenario_execution_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # Shader Execution Tests
    add_executable(shader_execution_tests
        integration/shader_execution/test_shader_execution.cpp
    )
    target_link_libraries(shader_execution_tests PRIVATE ${COMMON_TEST_LIBS} ${SPIRV_LIBS})
    
    # End-to-End Inference Tests
    add_executable(inference_tests
        integration/inference/test_inference.cpp
    )
    target_link_libraries(inference_tests PRIVATE ${COMMON_TEST_LIBS})
endif()

# === PERFORMANCE TESTS ===
if(BUILD_PERFORMANCE_TESTS)
    # Comprehensive Benchmark Suite
    add_executable(comprehensive_benchmarks
        performance/comprehensive_benchmarks.cpp
    )
    target_link_libraries(comprehensive_benchmarks PRIVATE ${COMMON_TEST_LIBS})
    target_compile_definitions(comprehensive_benchmarks PRIVATE ENABLE_BENCHMARKS)
    
    # Operation Benchmarks
    add_executable(operation_benchmarks
        performance/operations/test_operation_benchmarks.cpp
    )
    target_link_libraries(operation_benchmarks PRIVATE ${COMMON_TEST_LIBS})
    
    # Memory Benchmarks
    add_executable(memory_benchmarks
        performance/memory/test_memory_benchmarks.cpp
    )
    target_link_libraries(memory_benchmarks PRIVATE ${COMMON_TEST_LIBS})
endif()

# === STRESS TESTS ===
if(BUILD_STRESS_TESTS)
    # Concurrent Execution Tests
    add_executable(concurrent_tests
        stress/concurrent/test_concurrent_execution.cpp
    )
    target_link_libraries(concurrent_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # Large Model Tests
    add_executable(large_model_tests
        stress/large_models/test_large_models.cpp
    )
    target_link_libraries(large_model_tests PRIVATE ${COMMON_TEST_LIBS})
    
    # Resource Limit Tests
    add_executable(resource_limit_tests
        stress/resource_limits/test_resource_limits.cpp
    )
    target_link_libraries(resource_limit_tests PRIVATE ${COMMON_TEST_LIBS})
endif()

# === PLATFORM-SPECIFIC TESTS ===
if(BUILD_PLATFORM_TESTS AND APPLE_SILICON)
    # Apple Silicon Optimization Tests
    add_executable(apple_silicon_tests
        platform/macos_arm64/test_apple_silicon.cpp
    )
    target_link_libraries(apple_silicon_tests PRIVATE ${COMMON_TEST_LIBS})
    target_compile_definitions(apple_silicon_tests PRIVATE APPLE_SILICON_TESTS)
endif()

# === SYSTEM TESTS ===
add_executable(build_system_tests
    system/test_build_system.cpp
)
target_link_libraries(build_system_tests PRIVATE ${COMMON_TEST_LIBS})

# === VALIDATION TESTS ===
add_executable(ml_ops_validation_tests
    validation/ml_operations/test_ml_ops_validation.cpp
)
target_link_libraries(ml_ops_validation_tests PRIVATE ${COMMON_TEST_LIBS})

# === REGRESSION TESTS ===
add_executable(regression_tests
    regression/test_regression_suite.cpp
)
target_link_libraries(regression_tests PRIVATE ${COMMON_TEST_LIBS})

# CTest configuration
enable_testing()

# Register all tests with CTest
set(ALL_TEST_TARGETS
    vulkan_core_tests
    spirv_shader_tests
    memory_management_tests
    ml_operations_tests
    vgf_core_tests
    vgf_simple_tests
    vgf_minimal_tests
    scenario_runner_tests
    build_system_tests
    ml_ops_validation_tests
    regression_tests
)

if(BUILD_INTEGRATION_TESTS)
    list(APPEND ALL_TEST_TARGETS
        model_pipeline_tests
        scenario_execution_tests
        shader_execution_tests
        inference_tests
    )
endif()

if(BUILD_PERFORMANCE_TESTS)
    list(APPEND ALL_TEST_TARGETS
        comprehensive_benchmarks
        operation_benchmarks
        memory_benchmarks
    )
endif()

if(BUILD_STRESS_TESTS)
    list(APPEND ALL_TEST_TARGETS
        concurrent_tests
        large_model_tests
        resource_limit_tests
    )
endif()

if(BUILD_PLATFORM_TESTS AND APPLE_SILICON)
    list(APPEND ALL_TEST_TARGETS apple_silicon_tests)
endif()

# Add tests to CTest
foreach(test_target ${ALL_TEST_TARGETS})
    add_test(NAME ${test_target} COMMAND ${test_target})
    
    # Set test properties
    set_tests_properties(${test_target} PROPERTIES
        TIMEOUT 300  # 5 minutes timeout per test
        ENVIRONMENT "DYLD_LIBRARY_PATH=/usr/local/lib:${SDK_LIB_DIR};VK_LAYER_PATH=${SDK_LIB_DIR}"
    )
    
    # Add labels for test organization
    if(${test_target} MATCHES "^vulkan_|^spirv_|^memory_|^ml_operations_|^vgf_|^scenario_runner_")
        set_property(TEST ${test_target} PROPERTY LABELS "unit")
    elseif(${test_target} MATCHES "^model_pipeline_|^scenario_execution_|^shader_execution_|^inference_")
        set_property(TEST ${test_target} PROPERTY LABELS "integration")
    elseif(${test_target} MATCHES "benchmarks")
        set_property(TEST ${test_target} PROPERTY LABELS "performance")
    elseif(${test_target} MATCHES "^concurrent_|^large_model_|^resource_limit_")
        set_property(TEST ${test_target} PROPERTY LABELS "stress")
    elseif(${test_target} MATCHES "^apple_silicon_")
        set_property(TEST ${test_target} PROPERTY LABELS "platform")
    elseif(${test_target} MATCHES "^build_system_")
        set_property(TEST ${test_target} PROPERTY LABELS "system")
    elseif(${test_target} MATCHES "validation")
        set_property(TEST ${test_target} PROPERTY LABELS "validation")
    elseif(${test_target} MATCHES "regression")
        set_property(TEST ${test_target} PROPERTY LABELS "regression")
    endif()
endforeach()

# Custom target to run all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -j ${CMAKE_BUILD_PARALLEL_LEVEL} --output-on-failure
    DEPENDS ${ALL_TEST_TARGETS}
    COMMENT "Running all tests..."
)

# Custom target to run tests by category
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L unit --output-on-failure
    COMMENT "Running unit tests..."
)

add_custom_target(run_integration_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L integration --output-on-failure
    COMMENT "Running integration tests..."
)

add_custom_target(run_performance_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L performance --output-on-failure
    COMMENT "Running performance tests..."
)

add_custom_target(run_stress_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L stress --output-on-failure
    COMMENT "Running stress tests..."
)

# Test coverage target
if(ENABLE_COVERAGE)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)
    
    if(LCOV AND GENHTML)
        add_custom_target(coverage
            COMMAND ${LCOV} --directory . --zerocounters
            COMMAND ${CMAKE_CTEST_COMMAND}
            COMMAND ${LCOV} --directory . --capture --output-file coverage.info
            COMMAND ${LCOV} --remove coverage.info '/usr/*' '*/googletest/*' --output-file coverage.filtered.info
            COMMAND ${GENHTML} coverage.filtered.info --output-directory coverage_report
            COMMENT "Generating code coverage report..."
        )
    endif()
endif()

# Installation rules
install(TARGETS ${ALL_TEST_TARGETS}
    RUNTIME DESTINATION bin/tests
)

install(DIRECTORY ${TEST_DATA_DIR}
    DESTINATION share/tests
)

# Test report generation
add_custom_target(generate_test_report
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/framework/generate_report.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating test report..."
)

# Print configuration summary
message(STATUS "===== Test Suite Configuration =====")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Unit Tests: ${BUILD_UNIT_TESTS}")
message(STATUS "Integration Tests: ${BUILD_INTEGRATION_TESTS}")
message(STATUS "Performance Tests: ${BUILD_PERFORMANCE_TESTS}")
message(STATUS "Stress Tests: ${BUILD_STRESS_TESTS}")
message(STATUS "Platform Tests: ${BUILD_PLATFORM_TESTS}")
message(STATUS "Code Coverage: ${ENABLE_COVERAGE}")
message(STATUS "Sanitizers: ${ENABLE_SANITIZERS}")
message(STATUS "Parallel Execution: ${PARALLEL_TEST_EXECUTION}")
if(APPLE_SILICON)
    message(STATUS "Apple Silicon Optimizations: ENABLED")
endif()
message(STATUS "====================================")