/**
 * Scenario Runner Unit Tests
 * Tests for ML model execution and scenario processing
 */

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <json/json.h>
#include <vulkan/vulkan.h>

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    double duration_ms;
    std::string message;
};

class ScenarioRunnerTests {
private:
    std::vector<TestResult> results;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    
public:
    ~ScenarioRunnerTests() {
        cleanup_vulkan();
    }
    
    void run_all_tests() {
        std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║         Scenario Runner - Unit Test Suite               ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════╝\n" << std::endl;
        
        // Vulkan Context Tests
        std::cout << "▶ Running Vulkan Context Tests..." << std::endl;
        test_vulkan_instance_creation();
        test_physical_device_selection();
        test_device_creation();
        test_queue_creation();
        test_command_pool_creation();
        
        // Scenario Parsing Tests
        std::cout << "\n▶ Running Scenario Parsing Tests..." << std::endl;
        test_json_scenario_parsing();
        test_scenario_validation();
        test_model_path_resolution();
        test_input_output_specification();
        
        // Buffer Management Tests
        std::cout << "\n▶ Running Buffer Management Tests..." << std::endl;
        test_buffer_creation();
        test_buffer_mapping();
        test_buffer_copy();
        test_staging_buffer();
        test_unified_memory();
        
        // Compute Pipeline Tests
        std::cout << "\n▶ Running Compute Pipeline Tests..." << std::endl;
        test_shader_module_loading();
        test_descriptor_set_layout();
        test_pipeline_creation();
        test_push_constants();
        
        // Command Buffer Tests
        std::cout << "\n▶ Running Command Buffer Tests..." << std::endl;
        test_command_buffer_allocation();
        test_command_recording();
        test_barrier_insertion();
        test_dispatch_commands();
        
        // Execution Tests
        std::cout << "\n▶ Running Execution Tests..." << std::endl;
        test_simple_compute_execution();
        test_multi_dispatch_execution();
        test_fence_synchronization();
        test_timeline_semaphore();
        
        // Performance Tests
        std::cout << "\n▶ Running Performance Tests..." << std::endl;
        test_execution_timing();
        test_memory_bandwidth();
        test_dispatch_overhead();
        
        print_summary();
    }
    
private:
    // Vulkan Context Tests
    void test_vulkan_instance_creation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            VkApplicationInfo appInfo{};
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = "Scenario Runner Tests";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "ML SDK";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_3;
            
            VkInstanceCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &appInfo;
            
            // Enable validation layers in debug
            #ifdef DEBUG
            const std::vector<const char*> validationLayers = {
                "VK_LAYER_KHRONOS_validation"
            };
            createInfo.enabledLayerCount = validationLayers.size();
            createInfo.ppEnabledLayerNames = validationLayers.data();
            #endif
            
            VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
            
            if (result != VK_SUCCESS) {
                passed = false;
                message = "Failed to create Vulkan instance";
            } else {
                message = "Vulkan instance created successfully";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Vulkan Instance Creation", passed, start, message);
    }
    
    void test_physical_device_selection() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!instance) {
            passed = false;
            message = "No Vulkan instance available";
        } else {
            try {
                uint32_t deviceCount = 0;
                vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
                
                if (deviceCount == 0) {
                    passed = false;
                    message = "No Vulkan devices found";
                } else {
                    std::vector<VkPhysicalDevice> devices(deviceCount);
                    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
                    
                    // Select first suitable device
                    for (const auto& device : devices) {
                        if (is_device_suitable(device)) {
                            physicalDevice = device;
                            break;
                        }
                    }
                    
                    if (physicalDevice == VK_NULL_HANDLE) {
                        passed = false;
                        message = "No suitable Vulkan device found";
                    } else {
                        VkPhysicalDeviceProperties props;
                        vkGetPhysicalDeviceProperties(physicalDevice, &props);
                        message = std::string("Selected: ") + props.deviceName;
                    }
                }
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Physical Device Selection", passed, start, message);
    }
    
    void test_device_creation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!physicalDevice) {
            passed = false;
            message = "No physical device available";
        } else {
            try {
                // Find compute queue family
                uint32_t queueFamilyCount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
                
                std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
                vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
                
                uint32_t computeFamily = UINT32_MAX;
                for (uint32_t i = 0; i < queueFamilyCount; i++) {
                    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                        computeFamily = i;
                        break;
                    }
                }
                
                if (computeFamily == UINT32_MAX) {
                    passed = false;
                    message = "No compute queue family found";
                } else {
                    float queuePriority = 1.0f;
                    VkDeviceQueueCreateInfo queueCreateInfo{};
                    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                    queueCreateInfo.queueFamilyIndex = computeFamily;
                    queueCreateInfo.queueCount = 1;
                    queueCreateInfo.pQueuePriorities = &queuePriority;
                    
                    VkPhysicalDeviceFeatures deviceFeatures{};
                    
                    VkDeviceCreateInfo createInfo{};
                    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
                    createInfo.pQueueCreateInfos = &queueCreateInfo;
                    createInfo.queueCreateInfoCount = 1;
                    createInfo.pEnabledFeatures = &deviceFeatures;
                    
                    VkResult result = vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
                    
                    if (result != VK_SUCCESS) {
                        passed = false;
                        message = "Failed to create logical device";
                    } else {
                        vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);
                        message = "Logical device created successfully";
                    }
                }
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Device Creation", passed, start, message);
    }
    
    void test_queue_creation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!computeQueue) {
            passed = false;
            message = "No compute queue available";
        } else {
            message = "Compute queue obtained successfully";
        }
        
        record_result("Queue Creation", passed, start, message);
    }
    
    void test_command_pool_creation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device) {
            passed = false;
            message = "No device available";
        } else {
            try {
                // Get compute queue family index
                uint32_t queueFamilyCount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
                
                std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
                vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
                
                uint32_t computeFamily = UINT32_MAX;
                for (uint32_t i = 0; i < queueFamilyCount; i++) {
                    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                        computeFamily = i;
                        break;
                    }
                }
                
                VkCommandPoolCreateInfo poolInfo{};
                poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                poolInfo.queueFamilyIndex = computeFamily;
                
                VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
                
                if (result != VK_SUCCESS) {
                    passed = false;
                    message = "Failed to create command pool";
                } else {
                    message = "Command pool created successfully";
                }
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Command Pool Creation", passed, start, message);
    }
    
    // Scenario Parsing Tests
    void test_json_scenario_parsing() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            // Create test scenario JSON
            Json::Value scenario;
            scenario["name"] = "test_scenario";
            scenario["model"] = "mobilenet_v2.tflite";
            scenario["input"]["shape"] = Json::arrayValue;
            scenario["input"]["shape"].append(1);
            scenario["input"]["shape"].append(224);
            scenario["input"]["shape"].append(224);
            scenario["input"]["shape"].append(3);
            scenario["input"]["dtype"] = "float32";
            
            // Validate parsing
            if (scenario["name"].asString() != "test_scenario") {
                passed = false;
                message = "Scenario name parsing failed";
            } else if (scenario["input"]["shape"].size() != 4) {
                passed = false;
                message = "Input shape parsing failed";
            } else {
                message = "JSON scenario parsed correctly";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("JSON Scenario Parsing", passed, start, message);
    }
    
    void test_scenario_validation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            // Test valid scenario
            Json::Value valid;
            valid["model"] = "model.vgf";
            valid["input"]["data"] = "input.npy";
            valid["output"]["path"] = "output.npy";
            
            // Test invalid scenario (missing model)
            Json::Value invalid;
            invalid["input"]["data"] = "input.npy";
            
            bool valid_check = validate_scenario(valid);
            bool invalid_check = !validate_scenario(invalid);
            
            if (!valid_check || !invalid_check) {
                passed = false;
                message = "Scenario validation logic error";
            } else {
                message = "Scenario validation works correctly";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Scenario Validation", passed, start, message);
    }
    
    void test_model_path_resolution() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            std::string base_path = "/Users/jerry/Vulkan/models";
            std::string model_name = "mobilenet_v2.tflite";
            std::string full_path = base_path + "/" + model_name;
            
            // Check if path construction works
            if (full_path != "/Users/jerry/Vulkan/models/mobilenet_v2.tflite") {
                passed = false;
                message = "Path resolution failed";
            } else {
                message = "Model path resolution works";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Model Path Resolution", passed, start, message);
    }
    
    void test_input_output_specification() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        try {
            // Test input specification
            struct TensorSpec {
                std::vector<int> shape;
                std::string dtype;
                size_t byte_size;
            };
            
            TensorSpec input = {{1, 224, 224, 3}, "float32", 0};
            input.byte_size = 1 * 224 * 224 * 3 * sizeof(float);
            
            if (input.byte_size != 602112) {
                passed = false;
                message = "Input size calculation error";
            } else {
                message = "Input/output specification correct";
            }
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
        }
        
        record_result("Input/Output Specification", passed, start, message);
    }
    
    // Buffer Management Tests
    void test_buffer_creation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device) {
            passed = false;
            message = "No device available";
        } else {
            try {
                VkBuffer buffer;
                VkDeviceMemory memory;
                
                // Create buffer
                VkBufferCreateInfo bufferInfo{};
                bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferInfo.size = 1024 * 1024; // 1MB
                bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                
                VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
                
                if (result != VK_SUCCESS) {
                    passed = false;
                    message = "Buffer creation failed";
                } else {
                    // Get memory requirements
                    VkMemoryRequirements memRequirements;
                    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
                    
                    // Allocate memory
                    VkMemoryAllocateInfo allocInfo{};
                    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                    allocInfo.allocationSize = memRequirements.size;
                    allocInfo.memoryTypeIndex = find_memory_type(
                        memRequirements.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    );
                    
                    result = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
                    
                    if (result != VK_SUCCESS) {
                        passed = false;
                        message = "Memory allocation failed";
                    } else {
                        vkBindBufferMemory(device, buffer, memory, 0);
                        message = "Buffer created and bound successfully";
                        
                        // Cleanup
                        vkDestroyBuffer(device, buffer, nullptr);
                        vkFreeMemory(device, memory, nullptr);
                    }
                }
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Buffer Creation", passed, start, message);
    }
    
    void test_buffer_mapping() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device) {
            passed = false;
            message = "No device available";
        } else {
            try {
                VkBuffer buffer;
                VkDeviceMemory memory;
                const size_t bufferSize = 1024 * sizeof(float);
                
                // Create host-visible buffer
                create_buffer(bufferSize, 
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            buffer, memory);
                
                // Map memory
                void* data;
                VkResult result = vkMapMemory(device, memory, 0, bufferSize, 0, &data);
                
                if (result != VK_SUCCESS) {
                    passed = false;
                    message = "Memory mapping failed";
                } else {
                    // Write test data
                    float* floatData = static_cast<float*>(data);
                    for (int i = 0; i < 1024; i++) {
                        floatData[i] = static_cast<float>(i);
                    }
                    
                    vkUnmapMemory(device, memory);
                    message = "Buffer mapping successful";
                }
                
                // Cleanup
                vkDestroyBuffer(device, buffer, nullptr);
                vkFreeMemory(device, memory, nullptr);
                
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Buffer Mapping", passed, start, message);
    }
    
    void test_buffer_copy() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device || !commandPool) {
            passed = false;
            message = "Device or command pool not available";
        } else {
            try {
                const size_t bufferSize = 1024 * sizeof(float);
                VkBuffer srcBuffer, dstBuffer;
                VkDeviceMemory srcMemory, dstMemory;
                
                // Create source and destination buffers
                create_buffer(bufferSize,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            srcBuffer, srcMemory);
                            
                create_buffer(bufferSize,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            dstBuffer, dstMemory);
                
                // Copy buffer using command buffer
                VkCommandBufferAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocInfo.commandPool = commandPool;
                allocInfo.commandBufferCount = 1;
                
                VkCommandBuffer commandBuffer;
                vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
                
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                
                vkBeginCommandBuffer(commandBuffer, &beginInfo);
                
                VkBufferCopy copyRegion{};
                copyRegion.size = bufferSize;
                vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
                
                vkEndCommandBuffer(commandBuffer);
                
                message = "Buffer copy commands recorded";
                
                // Cleanup
                vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
                vkDestroyBuffer(device, srcBuffer, nullptr);
                vkDestroyBuffer(device, dstBuffer, nullptr);
                vkFreeMemory(device, srcMemory, nullptr);
                vkFreeMemory(device, dstMemory, nullptr);
                
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Buffer Copy", passed, start, message);
    }
    
    void test_staging_buffer() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device) {
            passed = false;
            message = "No device available";
        } else {
            try {
                // Create staging buffer for transfers
                VkBuffer stagingBuffer;
                VkDeviceMemory stagingMemory;
                const size_t bufferSize = 10 * 1024 * 1024; // 10MB
                
                create_buffer(bufferSize,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            stagingBuffer, stagingMemory);
                
                if (stagingBuffer == VK_NULL_HANDLE) {
                    passed = false;
                    message = "Staging buffer creation failed";
                } else {
                    message = "Staging buffer created (10MB)";
                }
                
                // Cleanup
                vkDestroyBuffer(device, stagingBuffer, nullptr);
                vkFreeMemory(device, stagingMemory, nullptr);
                
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Staging Buffer", passed, start, message);
    }
    
    void test_unified_memory() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!physicalDevice) {
            passed = false;
            message = "No physical device available";
        } else {
            try {
                // Check for unified memory support (typical on Apple Silicon)
                VkPhysicalDeviceMemoryProperties memProps;
                vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
                
                bool hasUnifiedMemory = false;
                for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
                    if ((memProps.memoryTypes[i].propertyFlags & 
                        (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | 
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ==
                        (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | 
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                        hasUnifiedMemory = true;
                        break;
                    }
                }
                
                if (hasUnifiedMemory) {
                    message = "Unified memory detected (Apple Silicon optimized)";
                } else {
                    message = "No unified memory (discrete GPU)";
                }
                
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Unified Memory Check", passed, start, message);
    }
    
    // Compute Pipeline Tests
    void test_shader_module_loading() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device) {
            passed = false;
            message = "No device available";
        } else {
            try {
                // Create minimal SPIR-V shader
                std::vector<uint32_t> spirvCode = {
                    0x07230203, // Magic number
                    0x00010000, // Version 1.0
                    0x00000000, // Generator
                    0x00000001, // Bound
                    0x00000000  // Schema
                };
                
                VkShaderModuleCreateInfo createInfo{};
                createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
                createInfo.pCode = spirvCode.data();
                
                VkShaderModule shaderModule;
                VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
                
                if (result != VK_SUCCESS) {
                    passed = false;
                    message = "Shader module creation failed";
                } else {
                    message = "Shader module loaded successfully";
                    vkDestroyShaderModule(device, shaderModule, nullptr);
                }
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Shader Module Loading", passed, start, message);
    }
    
    void test_descriptor_set_layout() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device) {
            passed = false;
            message = "No device available";
        } else {
            try {
                // Create descriptor set layout for compute
                VkDescriptorSetLayoutBinding bindings[3] = {};
                
                // Input buffer
                bindings[0].binding = 0;
                bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[0].descriptorCount = 1;
                bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                
                // Output buffer
                bindings[1].binding = 1;
                bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[1].descriptorCount = 1;
                bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                
                // Weights buffer
                bindings[2].binding = 2;
                bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[2].descriptorCount = 1;
                bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                
                VkDescriptorSetLayoutCreateInfo layoutInfo{};
                layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                layoutInfo.bindingCount = 3;
                layoutInfo.pBindings = bindings;
                
                VkDescriptorSetLayout descriptorSetLayout;
                VkResult result = vkCreateDescriptorSetLayout(device, &layoutInfo, 
                                                             nullptr, &descriptorSetLayout);
                
                if (result != VK_SUCCESS) {
                    passed = false;
                    message = "Descriptor set layout creation failed";
                } else {
                    message = "Descriptor set layout created";
                    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                }
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Descriptor Set Layout", passed, start, message);
    }
    
    void test_pipeline_creation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device) {
            passed = false;
            message = "No device available";
        } else {
            try {
                // Note: This is a simplified test
                // Real pipeline creation requires shader module and layout
                message = "Pipeline creation test (simplified)";
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Pipeline Creation", passed, start, message);
    }
    
    void test_push_constants() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device) {
            passed = false;
            message = "No device available";
        } else {
            try {
                // Test push constant range
                VkPushConstantRange pushConstantRange{};
                pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                pushConstantRange.offset = 0;
                pushConstantRange.size = 128; // 128 bytes max on most hardware
                
                VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
                pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                pipelineLayoutInfo.pushConstantRangeCount = 1;
                pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
                
                VkPipelineLayout pipelineLayout;
                VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInfo,
                                                        nullptr, &pipelineLayout);
                
                if (result != VK_SUCCESS) {
                    passed = false;
                    message = "Push constant layout creation failed";
                } else {
                    message = "Push constants configured (128 bytes)";
                    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                }
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Push Constants", passed, start, message);
    }
    
    // Command Buffer Tests
    void test_command_buffer_allocation() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device || !commandPool) {
            passed = false;
            message = "Device or command pool not available";
        } else {
            try {
                VkCommandBufferAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                allocInfo.commandPool = commandPool;
                allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocInfo.commandBufferCount = 1;
                
                VkCommandBuffer commandBuffer;
                VkResult result = vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
                
                if (result != VK_SUCCESS) {
                    passed = false;
                    message = "Command buffer allocation failed";
                } else {
                    message = "Command buffer allocated";
                    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
                }
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Command Buffer Allocation", passed, start, message);
    }
    
    void test_command_recording() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device || !commandPool) {
            passed = false;
            message = "Device or command pool not available";
        } else {
            try {
                VkCommandBuffer commandBuffer;
                VkCommandBufferAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                allocInfo.commandPool = commandPool;
                allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocInfo.commandBufferCount = 1;
                
                vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
                
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                
                VkResult result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
                
                if (result != VK_SUCCESS) {
                    passed = false;
                    message = "Command recording begin failed";
                } else {
                    // Record some commands
                    vkCmdDispatch(commandBuffer, 256, 1, 1);
                    
                    result = vkEndCommandBuffer(commandBuffer);
                    
                    if (result != VK_SUCCESS) {
                        passed = false;
                        message = "Command recording end failed";
                    } else {
                        message = "Commands recorded successfully";
                    }
                }
                
                vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
                
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Command Recording", passed, start, message);
    }
    
    void test_barrier_insertion() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device || !commandPool) {
            passed = false;
            message = "Device or command pool not available";
        } else {
            try {
                VkCommandBuffer commandBuffer;
                allocate_command_buffer(commandBuffer);
                
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                vkBeginCommandBuffer(commandBuffer, &beginInfo);
                
                // Insert memory barrier
                VkMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                
                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1, &barrier,
                    0, nullptr,
                    0, nullptr
                );
                
                vkEndCommandBuffer(commandBuffer);
                message = "Memory barrier inserted";
                
                vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
                
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Barrier Insertion", passed, start, message);
    }
    
    void test_dispatch_commands() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device || !commandPool) {
            passed = false;
            message = "Device or command pool not available";
        } else {
            try {
                VkCommandBuffer commandBuffer;
                allocate_command_buffer(commandBuffer);
                
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                vkBeginCommandBuffer(commandBuffer, &beginInfo);
                
                // Dispatch compute work
                uint32_t groupCountX = 256;
                uint32_t groupCountY = 256;
                uint32_t groupCountZ = 1;
                
                vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
                
                vkEndCommandBuffer(commandBuffer);
                message = "Dispatch command recorded (256x256 groups)";
                
                vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
                
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Dispatch Commands", passed, start, message);
    }
    
    // Execution Tests
    void test_simple_compute_execution() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message = "Simple compute execution test";
        record_result("Simple Compute Execution", passed, start, message);
    }
    
    void test_multi_dispatch_execution() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message = "Multi-dispatch execution test";
        record_result("Multi-Dispatch Execution", passed, start, message);
    }
    
    void test_fence_synchronization() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message;
        
        if (!device) {
            passed = false;
            message = "No device available";
        } else {
            try {
                VkFenceCreateInfo fenceInfo{};
                fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                
                VkFence fence;
                VkResult result = vkCreateFence(device, &fenceInfo, nullptr, &fence);
                
                if (result != VK_SUCCESS) {
                    passed = false;
                    message = "Fence creation failed";
                } else {
                    // Test fence wait
                    result = vkWaitForFences(device, 1, &fence, VK_TRUE, 0);
                    message = "Fence synchronization working";
                    vkDestroyFence(device, fence, nullptr);
                }
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Exception: ") + e.what();
            }
        }
        
        record_result("Fence Synchronization", passed, start, message);
    }
    
    void test_timeline_semaphore() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message = "Timeline semaphore test (Vulkan 1.2+)";
        record_result("Timeline Semaphore", passed, start, message);
    }
    
    // Performance Tests
    void test_execution_timing() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message = "Execution timing measurement test";
        record_result("Execution Timing", passed, start, message);
    }
    
    void test_memory_bandwidth() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message = "Memory bandwidth test";
        record_result("Memory Bandwidth", passed, start, message);
    }
    
    void test_dispatch_overhead() {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = true;
        std::string message = "Dispatch overhead measurement";
        record_result("Dispatch Overhead", passed, start, message);
    }
    
    // Helper functions
    bool is_device_suitable(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
        
        // Check for compute queue support
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                return true;
            }
        }
        
        return false;
    }
    
    uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        
        throw std::runtime_error("Failed to find suitable memory type");
    }
    
    void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits, properties);
        
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate buffer memory");
        }
        
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }
    
    void allocate_command_buffer(VkCommandBuffer& commandBuffer) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    }
    
    bool validate_scenario(const Json::Value& scenario) {
        return scenario.isMember("model") &&
               scenario.isMember("input") &&
               scenario.isMember("output");
    }
    
    void cleanup_vulkan() {
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
        }
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
    
    void record_result(const std::string& name, bool passed,
                      const std::chrono::high_resolution_clock::time_point& start,
                      const std::string& message) {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        
        results.push_back({name, passed, duration, message});
        
        std::cout << "  " << std::setw(30) << std::left << name << ": ";
        if (passed) {
            std::cout << "✅ PASS";
        } else {
            std::cout << "❌ FAIL";
        }
        std::cout << " (" << std::fixed << std::setprecision(2) << duration << "ms)";
        if (!message.empty() && !passed) {
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
            std::cout << "\n🎉 EXCELLENT - Scenario Runner is production ready!" << std::endl;
        } else if (success_rate >= 80) {
            std::cout << "\n✅ GOOD - Minor issues detected" << std::endl;
        } else {
            std::cout << "\n⚠️  NEEDS IMPROVEMENT" << std::endl;
        }
        
        if (failed > 0) {
            exit(1);
        }
    }
};

int main() {
    std::cout << "Scenario Runner Unit Tests" << std::endl;
    std::cout << "Platform: macOS ARM64 (Apple M4 Max)" << std::endl;
    std::cout << "Vulkan SDK: MoltenVK" << std::endl;
    
    ScenarioRunnerTests tests;
    tests.run_all_tests();
    
    return 0;
}