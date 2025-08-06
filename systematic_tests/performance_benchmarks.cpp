// Systematic Performance Benchmarks for ARM ML SDK
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <random>
#include <thread>

#define CYAN "\033[0;36m"
#define GREEN "\033[0;32m"
#define YELLOW "\033[0;33m"
#define MAGENTA "\033[0;35m"
#define NC "\033[0m"

class PerformanceBenchmark {
private:
    struct BenchmarkResult {
        std::string name;
        double min_ms;
        double max_ms;
        double avg_ms;
        double stddev_ms;
        double throughput;
        std::string unit;
    };
    
    std::vector<BenchmarkResult> results;
    std::mt19937 rng{42}; // Fixed seed for reproducibility
    
    template<typename Func>
    BenchmarkResult measure(const std::string& name, int iterations, Func f, 
                           double ops_count = 0, const std::string& unit = "") {
        std::vector<double> times;
        times.reserve(iterations);
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            f();
        }
        
        // Actual measurements
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            f();
            auto end = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(ms);
        }
        
        // Calculate statistics
        double min = *std::min_element(times.begin(), times.end());
        double max = *std::max_element(times.begin(), times.end());
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        double sq_sum = 0;
        for (double t : times) {
            sq_sum += (t - avg) * (t - avg);
        }
        double stddev = std::sqrt(sq_sum / times.size());
        
        double throughput = 0;
        if (ops_count > 0 && avg > 0) {
            throughput = ops_count / (avg / 1000.0); // ops per second
        }
        
        return {name, min, max, avg, stddev, throughput, unit};
    }
    
public:
    // 1. Memory Bandwidth Benchmark
    void benchmark_memory_bandwidth() {
        std::cout << CYAN << "MEMORY BANDWIDTH BENCHMARKS" << NC << std::endl;
        
        // Sequential read
        const size_t SIZE = 100 * 1024 * 1024; // 100MB
        std::vector<float> data(SIZE / sizeof(float));
        std::fill(data.begin(), data.end(), 1.0f);
        
        auto result = measure("Sequential Read (100MB)", 50, [&]() {
            float sum = 0;
            for (size_t i = 0; i < data.size(); i++) {
                sum += data[i];
            }
            volatile float prevent_opt = sum; // Prevent optimization
        }, SIZE, "bytes");
        
        results.push_back(result);
        std::cout << "  Read Bandwidth: " << std::fixed << std::setprecision(2) 
                  << result.throughput / (1024*1024*1024) << " GB/s" << std::endl;
        
        // Sequential write
        result = measure("Sequential Write (100MB)", 50, [&]() {
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = static_cast<float>(i);
            }
        }, SIZE, "bytes");
        
        results.push_back(result);
        std::cout << "  Write Bandwidth: " << result.throughput / (1024*1024*1024) << " GB/s" << std::endl;
        
        // Copy bandwidth
        std::vector<float> dest(data.size());
        result = measure("Memory Copy (100MB)", 50, [&]() {
            std::copy(data.begin(), data.end(), dest.begin());
        }, SIZE * 2, "bytes"); // Read + Write
        
        results.push_back(result);
        std::cout << "  Copy Bandwidth: " << result.throughput / (1024*1024*1024) << " GB/s" << std::endl;
        std::cout << std::endl;
    }
    
    // 2. Compute Benchmarks
    void benchmark_compute_operations() {
        std::cout << CYAN << "COMPUTE OPERATION BENCHMARKS" << NC << std::endl;
        
        // Vector operations
        const int VECTOR_SIZE = 1024 * 1024; // 1M elements
        std::vector<float> a(VECTOR_SIZE), b(VECTOR_SIZE), c(VECTOR_SIZE);
        
        // Initialize with random data
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < VECTOR_SIZE; i++) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
        
        // Vector Addition
        auto result = measure("Vector Add (1M elements)", 100, [&]() {
            for (int i = 0; i < VECTOR_SIZE; i++) {
                c[i] = a[i] + b[i];
            }
        }, VECTOR_SIZE * 2, "FLOPS");
        
        results.push_back(result);
        std::cout << "  Vector Add: " << result.avg_ms << "ms (" 
                  << result.throughput / 1e9 << " GFLOPS)" << std::endl;
        
        // Vector Multiply-Add (FMA)
        result = measure("Vector FMA (1M elements)", 100, [&]() {
            for (int i = 0; i < VECTOR_SIZE; i++) {
                c[i] = a[i] * b[i] + c[i];
            }
        }, VECTOR_SIZE * 3, "FLOPS");
        
        results.push_back(result);
        std::cout << "  Vector FMA: " << result.avg_ms << "ms (" 
                  << result.throughput / 1e9 << " GFLOPS)" << std::endl;
        
        // Dot Product
        result = measure("Dot Product (1M elements)", 100, [&]() {
            float sum = 0;
            for (int i = 0; i < VECTOR_SIZE; i++) {
                sum += a[i] * b[i];
            }
            volatile float prevent_opt = sum;
        }, VECTOR_SIZE * 2, "FLOPS");
        
        results.push_back(result);
        std::cout << "  Dot Product: " << result.avg_ms << "ms (" 
                  << result.throughput / 1e9 << " GFLOPS)" << std::endl;
        std::cout << std::endl;
    }
    
    // 3. Matrix Operations
    void benchmark_matrix_operations() {
        std::cout << CYAN << "MATRIX OPERATION BENCHMARKS" << NC << std::endl;
        
        // Matrix sizes
        const int sizes[] = {128, 256, 512, 1024};
        
        for (int N : sizes) {
            std::vector<float> A(N * N), B(N * N), C(N * N);
            
            // Initialize
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int i = 0; i < N * N; i++) {
                A[i] = dist(rng);
                B[i] = dist(rng);
            }
            
            // Matrix Multiply
            auto result = measure("MatMul " + std::to_string(N) + "x" + std::to_string(N), 
                                10, [&]() {
                // Simple triple loop (not optimized)
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        float sum = 0;
                        for (int k = 0; k < N; k++) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }, 2.0 * N * N * N, "FLOPS");
            
            results.push_back(result);
            std::cout << "  MatMul " << N << "x" << N << ": " 
                      << std::fixed << std::setprecision(2) << result.avg_ms << "ms (" 
                      << result.throughput / 1e9 << " GFLOPS)" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 4. ML-specific Operations
    void benchmark_ml_operations() {
        std::cout << CYAN << "ML OPERATION BENCHMARKS" << NC << std::endl;
        
        // Convolution benchmark (simplified)
        const int INPUT_SIZE = 224;
        const int KERNEL_SIZE = 3;
        const int OUTPUT_SIZE = INPUT_SIZE - KERNEL_SIZE + 1;
        const int CHANNELS = 32;
        
        std::vector<float> input(INPUT_SIZE * INPUT_SIZE * CHANNELS);
        std::vector<float> kernel(KERNEL_SIZE * KERNEL_SIZE * CHANNELS);
        std::vector<float> output(OUTPUT_SIZE * OUTPUT_SIZE);
        
        // Initialize
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : input) v = dist(rng);
        for (auto& v : kernel) v = dist(rng);
        
        auto result = measure("Conv2D (224x224x32, 3x3)", 10, [&]() {
            // Simplified convolution
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                for (int x = 0; x < OUTPUT_SIZE; x++) {
                    float sum = 0;
                    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                            for (int c = 0; c < CHANNELS; c++) {
                                int in_idx = ((y + ky) * INPUT_SIZE + (x + kx)) * CHANNELS + c;
                                int k_idx = (ky * KERNEL_SIZE + kx) * CHANNELS + c;
                                sum += input[in_idx] * kernel[k_idx];
                            }
                        }
                    }
                    output[y * OUTPUT_SIZE + x] = sum;
                }
            }
        }, OUTPUT_SIZE * OUTPUT_SIZE * KERNEL_SIZE * KERNEL_SIZE * CHANNELS * 2, "FLOPS");
        
        results.push_back(result);
        std::cout << "  Conv2D: " << result.avg_ms << "ms (" 
                  << result.throughput / 1e9 << " GFLOPS)" << std::endl;
        
        // ReLU activation
        const int ACTIVATION_SIZE = 1024 * 1024;
        std::vector<float> act_data(ACTIVATION_SIZE);
        for (auto& v : act_data) v = dist(rng) - 0.5f; // Some negative values
        
        result = measure("ReLU (1M elements)", 100, [&]() {
            for (int i = 0; i < ACTIVATION_SIZE; i++) {
                act_data[i] = std::max(0.0f, act_data[i]);
            }
        }, ACTIVATION_SIZE, "ops");
        
        results.push_back(result);
        std::cout << "  ReLU: " << result.avg_ms << "ms (" 
                  << result.throughput / 1e6 << "M ops/sec)" << std::endl;
        
        // MaxPool 2x2
        const int POOL_INPUT = 256;
        const int POOL_OUTPUT = POOL_INPUT / 2;
        std::vector<float> pool_input(POOL_INPUT * POOL_INPUT);
        std::vector<float> pool_output(POOL_OUTPUT * POOL_OUTPUT);
        
        for (auto& v : pool_input) v = dist(rng);
        
        result = measure("MaxPool2x2 (256x256)", 100, [&]() {
            for (int y = 0; y < POOL_OUTPUT; y++) {
                for (int x = 0; x < POOL_OUTPUT; x++) {
                    float max_val = pool_input[(y*2) * POOL_INPUT + (x*2)];
                    max_val = std::max(max_val, pool_input[(y*2) * POOL_INPUT + (x*2+1)]);
                    max_val = std::max(max_val, pool_input[(y*2+1) * POOL_INPUT + (x*2)]);
                    max_val = std::max(max_val, pool_input[(y*2+1) * POOL_INPUT + (x*2+1)]);
                    pool_output[y * POOL_OUTPUT + x] = max_val;
                }
            }
        }, POOL_OUTPUT * POOL_OUTPUT * 4, "comparisons");
        
        results.push_back(result);
        std::cout << "  MaxPool2x2: " << result.avg_ms << "ms" << std::endl;
        std::cout << std::endl;
    }
    
    // 5. Parallel Processing
    void benchmark_parallel_processing() {
        std::cout << CYAN << "PARALLEL PROCESSING BENCHMARKS" << NC << std::endl;
        
        const int SIZE = 10 * 1024 * 1024; // 10M elements
        std::vector<float> data(SIZE);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : data) v = dist(rng);
        
        // Single-threaded
        auto result = measure("Single-threaded Sum (10M)", 10, [&]() {
            float sum = 0;
            for (int i = 0; i < SIZE; i++) {
                sum += data[i];
            }
            volatile float prevent_opt = sum;
        }, SIZE, "ops");
        
        results.push_back(result);
        std::cout << "  Single-threaded: " << result.avg_ms << "ms" << std::endl;
        
        // Multi-threaded
        int num_threads = std::thread::hardware_concurrency();
        result = measure("Multi-threaded Sum (10M)", 10, [&]() {
            std::vector<std::thread> threads;
            std::vector<float> partial_sums(num_threads, 0);
            int chunk_size = SIZE / num_threads;
            
            for (int t = 0; t < num_threads; t++) {
                threads.emplace_back([&, t]() {
                    int start = t * chunk_size;
                    int end = (t == num_threads - 1) ? SIZE : start + chunk_size;
                    float sum = 0;
                    for (int i = start; i < end; i++) {
                        sum += data[i];
                    }
                    partial_sums[t] = sum;
                });
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
            
            float total = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0f);
            volatile float prevent_opt = total;
        }, SIZE, "ops");
        
        results.push_back(result);
        std::cout << "  Multi-threaded (" << num_threads << " threads): " 
                  << result.avg_ms << "ms" << std::endl;
        
        float speedup = results[results.size()-2].avg_ms / results[results.size()-1].avg_ms;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << std::endl;
    }
    
    void print_summary() {
        std::cout << MAGENTA << "═══════════════════════════════════════════════════════════" << NC << std::endl;
        std::cout << MAGENTA << "                  PERFORMANCE SUMMARY" << NC << std::endl;
        std::cout << MAGENTA << "═══════════════════════════════════════════════════════════" << NC << std::endl;
        std::cout << std::endl;
        
        std::cout << std::left << std::setw(30) << "Operation" 
                  << std::right << std::setw(10) << "Min (ms)"
                  << std::setw(10) << "Avg (ms)"
                  << std::setw(10) << "Max (ms)"
                  << std::setw(15) << "Throughput" << std::endl;
        std::cout << std::string(75, '-') << std::endl;
        
        for (const auto& r : results) {
            std::cout << std::left << std::setw(30) << r.name
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(10) << r.min_ms
                      << std::setw(10) << r.avg_ms
                      << std::setw(10) << r.max_ms;
            
            if (r.throughput > 1e9) {
                std::cout << std::setw(12) << (r.throughput / 1e9) << " G" << r.unit << "/s";
            } else if (r.throughput > 1e6) {
                std::cout << std::setw(12) << (r.throughput / 1e6) << " M" << r.unit << "/s";
            } else if (r.throughput > 0) {
                std::cout << std::setw(12) << r.throughput << " " << r.unit << "/s";
            }
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << GREEN << "✅ Performance benchmarks complete!" << NC << std::endl;
    }
    
    void run_all() {
        benchmark_memory_bandwidth();
        benchmark_compute_operations();
        benchmark_matrix_operations();
        benchmark_ml_operations();
        benchmark_parallel_processing();
        print_summary();
    }
};

int main() {
    std::cout << MAGENTA << "═══════════════════════════════════════════════════════════" << NC << std::endl;
    std::cout << MAGENTA << "     ARM ML SDK - Performance Benchmarks" << NC << std::endl;
    std::cout << MAGENTA << "═══════════════════════════════════════════════════════════" << NC << std::endl;
    std::cout << std::endl;
    
    PerformanceBenchmark benchmark;
    benchmark.run_all();
    
    return 0;
}