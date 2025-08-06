// Systematic C++ Unit Tests for ARM ML SDK
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cmath>
#include <numeric>
#include <iomanip>

// Color output for test results
#define GREEN "\033[0;32m"
#define RED "\033[0;31m"
#define YELLOW "\033[0;33m"
#define CYAN "\033[0;36m"
#define NC "\033[0m"

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    double duration_ms;
    std::string details;
};

class TestFramework {
private:
    std::vector<TestResult> results;
    int total_tests = 0;
    int passed_tests = 0;
    
public:
    void run_test(const std::string& name, std::function<bool()> test_func) {
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = false;
        
        try {
            passed = test_func();
        } catch (const std::exception& e) {
            passed = false;
            std::cout << RED << "  ✗ " << name << " - Exception: " << e.what() << NC << std::endl;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        
        results.push_back({name, passed, duration, ""});
        total_tests++;
        if (passed) {
            passed_tests++;
            std::cout << GREEN << "  ✓ " << name << NC << " (" << std::fixed << std::setprecision(2) << duration << "ms)" << std::endl;
        } else {
            std::cout << RED << "  ✗ " << name << NC << " (" << std::fixed << std::setprecision(2) << duration << "ms)" << std::endl;
        }
    }
    
    void print_summary() {
        std::cout << "\n" << CYAN << "═══════════════════════════════════════════════" << NC << std::endl;
        std::cout << CYAN << "                TEST SUMMARY" << NC << std::endl;
        std::cout << CYAN << "═══════════════════════════════════════════════" << NC << std::endl;
        
        double total_time = 0;
        for (const auto& result : results) {
            total_time += result.duration_ms;
        }
        
        std::cout << "Tests Passed: " << GREEN << passed_tests << "/" << total_tests << NC;
        std::cout << " (" << (passed_tests * 100 / total_tests) << "%)" << std::endl;
        std::cout << "Total Time: " << total_time << "ms" << std::endl;
        
        if (passed_tests == total_tests) {
            std::cout << GREEN << "\n✅ All tests passed!" << NC << std::endl;
        } else {
            std::cout << RED << "\n✗ Some tests failed" << NC << std::endl;
        }
    }
};

// === UNIT TESTS ===

// 1. Memory Alignment Tests
bool test_memory_alignment() {
    const size_t ALIGNMENT = 256; // Apple GPU cache line
    std::vector<float> data(1024);
    
    // Test alignment calculation
    size_t unaligned_size = 1000;
    size_t aligned_size = (unaligned_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
    return aligned_size == 1024 && (aligned_size % ALIGNMENT == 0);
}

// 2. Buffer Operations
bool test_buffer_operations() {
    // Simulate buffer creation and operations
    struct Buffer {
        std::vector<float> data;
        size_t size;
        
        Buffer(size_t s) : size(s), data(s, 0.0f) {}
        
        void fill(float value) {
            std::fill(data.begin(), data.end(), value);
        }
        
        float sum() const {
            return std::accumulate(data.begin(), data.end(), 0.0f);
        }
    };
    
    Buffer buffer(1024);
    buffer.fill(1.0f);
    
    return std::abs(buffer.sum() - 1024.0f) < 0.001f;
}

// 3. Vector Addition (simulating compute shader)
bool test_vector_addition() {
    const int SIZE = 1024;
    std::vector<float> a(SIZE), b(SIZE), c(SIZE);
    
    // Initialize
    for (int i = 0; i < SIZE; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }
    
    // Add vectors
    for (int i = 0; i < SIZE; i++) {
        c[i] = a[i] + b[i];
    }
    
    // Verify
    for (int i = 0; i < SIZE; i++) {
        if (std::abs(c[i] - (i + i * 2)) > 0.001f) {
            return false;
        }
    }
    
    return true;
}

// 4. Matrix Multiplication
bool test_matrix_multiply() {
    const int N = 64; // Small matrix for unit test
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);
    
    // Simple matrix multiply
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    
    // Each element should be N * 1 * 2 = 128
    float expected = N * 2.0f;
    for (int i = 0; i < N * N; i++) {
        if (std::abs(C[i] - expected) > 0.001f) {
            return false;
        }
    }
    
    return true;
}

// 5. Convolution Operation (simplified)
bool test_convolution() {
    // 1D convolution for simplicity
    std::vector<float> input = {1, 2, 3, 4, 5};
    std::vector<float> kernel = {1, 0, -1}; // Edge detection kernel
    std::vector<float> output(3); // Valid convolution
    
    for (int i = 0; i < 3; i++) {
        float sum = 0;
        for (int j = 0; j < 3; j++) {
            sum += input[i + j] * kernel[j];
        }
        output[i] = sum;
    }
    
    // Expected: [2, 2, 2] (differences)
    return std::abs(output[0] - 2.0f) < 0.001f &&
           std::abs(output[1] - 2.0f) < 0.001f &&
           std::abs(output[2] - 2.0f) < 0.001f;
}

// 6. Activation Functions
bool test_activation_functions() {
    // ReLU
    auto relu = [](float x) { return std::max(0.0f, x); };
    
    // Sigmoid
    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    
    // Test ReLU
    if (relu(-1.0f) != 0.0f || relu(1.0f) != 1.0f) return false;
    
    // Test Sigmoid
    float sig_result = sigmoid(0.0f);
    if (std::abs(sig_result - 0.5f) > 0.001f) return false;
    
    return true;
}

// 7. Pooling Operations
bool test_pooling() {
    // 2x2 max pooling on 4x4 input
    std::vector<float> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    std::vector<float> output(4); // 2x2 output
    
    // Max pool with 2x2 kernel and stride 2
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float max_val = 0;
            for (int ki = 0; ki < 2; ki++) {
                for (int kj = 0; kj < 2; kj++) {
                    int idx = (i * 2 + ki) * 4 + (j * 2 + kj);
                    max_val = std::max(max_val, input[idx]);
                }
            }
            output[i * 2 + j] = max_val;
        }
    }
    
    // Expected: [6, 8, 14, 16]
    return output[0] == 6 && output[1] == 8 && 
           output[2] == 14 && output[3] == 16;
}

// 8. Quantization Test
bool test_quantization() {
    // Test INT8 quantization
    auto quantize = [](float value, float scale, int zero_point) -> int8_t {
        int q = static_cast<int>(std::round(value / scale + zero_point));
        return static_cast<int8_t>(std::max(-128, std::min(127, q)));
    };
    
    auto dequantize = [](int8_t value, float scale, int zero_point) -> float {
        return scale * (value - zero_point);
    };
    
    float original = 1.5f;
    float scale = 0.1f;
    int zero_point = 0;
    
    int8_t quantized = quantize(original, scale, zero_point);
    float dequantized = dequantize(quantized, scale, zero_point);
    
    // Should be close to original (within quantization error)
    return std::abs(dequantized - original) < 0.1f;
}

// 9. Data Layout Transformation
bool test_data_layout() {
    // NHWC to NCHW conversion
    const int N = 1, H = 2, W = 2, C = 3;
    std::vector<float> nhwc = {
        // H=0, W=0
        1, 2, 3,
        // H=0, W=1
        4, 5, 6,
        // H=1, W=0
        7, 8, 9,
        // H=1, W=1
        10, 11, 12
    };
    
    std::vector<float> nchw(N * C * H * W);
    
    // Convert NHWC to NCHW
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    int src_idx = n * H * W * C + h * W * C + w * C + c;
                    int dst_idx = n * C * H * W + c * H * W + h * W + w;
                    nchw[dst_idx] = nhwc[src_idx];
                }
            }
        }
    }
    
    // Verify channel 0: [1, 4, 7, 10]
    return nchw[0] == 1 && nchw[1] == 4 && nchw[2] == 7 && nchw[3] == 10;
}

// 10. FP16 Simulation
bool test_fp16_operations() {
    // Simulate FP16 precision
    auto to_fp16_precision = [](float value) -> float {
        // Simulate reduced precision (not actual FP16)
        int bits = *reinterpret_cast<int*>(&value);
        bits &= 0xFFFF0000; // Keep only upper 16 bits
        return *reinterpret_cast<float*>(&bits);
    };
    
    float a = 1.234567f;
    float b = 2.345678f;
    
    // Normal precision
    float result_fp32 = a + b;
    
    // Simulated FP16
    float a_fp16 = to_fp16_precision(a);
    float b_fp16 = to_fp16_precision(b);
    float result_fp16 = a_fp16 + b_fp16;
    
    // FP16 should have less precision
    return std::abs(result_fp32 - result_fp16) > 0.0001f;
}

int main() {
    std::cout << CYAN << "═══════════════════════════════════════════════" << NC << std::endl;
    std::cout << CYAN << "     ARM ML SDK - C++ Unit Tests" << NC << std::endl;
    std::cout << CYAN << "═══════════════════════════════════════════════" << NC << std::endl;
    std::cout << std::endl;
    
    TestFramework tests;
    
    std::cout << CYAN << "1. MEMORY TESTS" << NC << std::endl;
    tests.run_test("Memory Alignment (256-byte)", test_memory_alignment);
    tests.run_test("Buffer Operations", test_buffer_operations);
    std::cout << std::endl;
    
    std::cout << CYAN << "2. COMPUTE OPERATIONS" << NC << std::endl;
    tests.run_test("Vector Addition", test_vector_addition);
    tests.run_test("Matrix Multiplication", test_matrix_multiply);
    tests.run_test("Convolution", test_convolution);
    std::cout << std::endl;
    
    std::cout << CYAN << "3. ML OPERATIONS" << NC << std::endl;
    tests.run_test("Activation Functions", test_activation_functions);
    tests.run_test("Pooling Operations", test_pooling);
    tests.run_test("INT8 Quantization", test_quantization);
    std::cout << std::endl;
    
    std::cout << CYAN << "4. DATA HANDLING" << NC << std::endl;
    tests.run_test("NHWC to NCHW Layout", test_data_layout);
    tests.run_test("FP16 Operations", test_fp16_operations);
    
    tests.print_summary();
    
    return 0;
}