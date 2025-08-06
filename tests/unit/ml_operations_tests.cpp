/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 * 
 * ML Operations Unit Tests for ARM ML SDK
 * Tests correctness of ML operations: convolution, matrix multiplication,
 * activation functions, pooling, batch normalization, and quantization.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <array>

namespace mlsdk::tests {

class MLOperationsTest : public ::testing::Test {
protected:
    std::mt19937 rng{42}; // Deterministic random number generator
    
    // Helper to compare floating point arrays with tolerance
    bool compareArrays(const std::vector<float>& a, const std::vector<float>& b, 
                      float tolerance = 1e-5f) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    // Generate random tensor
    std::vector<float> generateRandomTensor(size_t size, float min = -1.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        std::vector<float> tensor(size);
        std::generate(tensor.begin(), tensor.end(), [&]() { return dist(rng); });
        return tensor;
    }
};

// Test 1: 2D Convolution
TEST_F(MLOperationsTest, Convolution2D) {
    // Simple 3x3 convolution on 5x5 input with 3x3 kernel
    const int inputH = 5, inputW = 5;
    const int kernelH = 3, kernelW = 3;
    const int outputH = inputH - kernelH + 1; // Valid convolution
    const int outputW = inputW - kernelW + 1;
    
    // Input: 5x5 matrix
    std::vector<float> input = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    
    // Kernel: 3x3 edge detection
    std::vector<float> kernel = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    
    // Compute convolution
    std::vector<float> output(outputH * outputW);
    for (int oh = 0; oh < outputH; ++oh) {
        for (int ow = 0; ow < outputW; ++ow) {
            float sum = 0.0f;
            for (int kh = 0; kh < kernelH; ++kh) {
                for (int kw = 0; kw < kernelW; ++kw) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    sum += input[ih * inputW + iw] * kernel[kh * kernelW + kw];
                }
            }
            output[oh * outputW + ow] = sum;
        }
    }
    
    // Expected output for center pixel: 13*8 - (7+8+9+12+14+17+18+19) = 104 - 104 = 0
    EXPECT_NEAR(output[4], 0.0f, 1e-5f); // Center of 3x3 output
}

// Test 2: Depthwise Convolution
TEST_F(MLOperationsTest, DepthwiseConvolution) {
    // 3x3 depthwise convolution with 2 channels
    const int channels = 2;
    const int inputH = 4, inputW = 4;
    const int kernelH = 3, kernelW = 3;
    const int outputH = inputH - kernelH + 1;
    const int outputW = inputW - kernelW + 1;
    
    // Input: 4x4x2 (HWC format)
    std::vector<float> input(inputH * inputW * channels);
    for (int i = 0; i < inputH * inputW; ++i) {
        input[i * channels + 0] = static_cast<float>(i);     // Channel 0
        input[i * channels + 1] = static_cast<float>(i * 2); // Channel 1
    }
    
    // Kernels: 3x3x2 (one per channel)
    std::vector<float> kernels(kernelH * kernelW * channels, 1.0f / 9.0f); // Average filter
    
    // Compute depthwise convolution
    std::vector<float> output(outputH * outputW * channels);
    for (int oh = 0; oh < outputH; ++oh) {
        for (int ow = 0; ow < outputW; ++ow) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int kh = 0; kh < kernelH; ++kh) {
                    for (int kw = 0; kw < kernelW; ++kw) {
                        int ih = oh + kh;
                        int iw = ow + kw;
                        int inputIdx = (ih * inputW + iw) * channels + c;
                        int kernelIdx = (kh * kernelW + kw) * channels + c;
                        sum += input[inputIdx] * kernels[kernelIdx];
                    }
                }
                output[(oh * outputW + ow) * channels + c] = sum;
            }
        }
    }
    
    // Verify output dimensions
    EXPECT_EQ(output.size(), outputH * outputW * channels);
    // Verify first output is average of first 3x3 region
    float expectedCh0 = (0 + 1 + 2 + 4 + 5 + 6 + 8 + 9 + 10) / 9.0f;
    EXPECT_NEAR(output[0], expectedCh0, 1e-5f);
}

// Test 3: Matrix Multiplication
TEST_F(MLOperationsTest, MatrixMultiplication) {
    // Multiply matrices: (2x3) x (3x4) = (2x4)
    const int M = 2, K = 3, N = 4;
    
    std::vector<float> A = {
        1, 2, 3,
        4, 5, 6
    };
    
    std::vector<float> B = {
        7, 8, 9, 10,
        11, 12, 13, 14,
        15, 16, 17, 18
    };
    
    std::vector<float> C(M * N);
    
    // Compute C = A * B
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
    
    // Expected: C[0,0] = 1*7 + 2*11 + 3*15 = 7 + 22 + 45 = 74
    EXPECT_NEAR(C[0], 74.0f, 1e-5f);
    // Expected: C[1,3] = 4*10 + 5*14 + 6*18 = 40 + 70 + 108 = 218
    EXPECT_NEAR(C[7], 218.0f, 1e-5f);
}

// Test 4: Batch Matrix Multiplication
TEST_F(MLOperationsTest, BatchMatMul) {
    const int batch = 2, M = 3, K = 4, N = 5;
    
    auto A = generateRandomTensor(batch * M * K);
    auto B = generateRandomTensor(batch * K * N);
    std::vector<float> C(batch * M * N);
    
    // Compute batch matmul
    for (int b = 0; b < batch; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int aIdx = b * M * K + m * K + k;
                    int bIdx = b * K * N + k * N + n;
                    sum += A[aIdx] * B[bIdx];
                }
                C[b * M * N + m * N + n] = sum;
            }
        }
    }
    
    // Verify output shape
    EXPECT_EQ(C.size(), batch * M * N);
    
    // Verify no NaN or Inf values
    for (float val : C) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

// Test 5: ReLU Activation
TEST_F(MLOperationsTest, ReLUActivation) {
    std::vector<float> input = {-2, -1, 0, 1, 2, 3, -0.5f, 0.5f};
    std::vector<float> output(input.size());
    
    // Apply ReLU: max(0, x)
    std::transform(input.begin(), input.end(), output.begin(),
                  [](float x) { return std::max(0.0f, x); });
    
    std::vector<float> expected = {0, 0, 0, 1, 2, 3, 0, 0.5f};
    EXPECT_TRUE(compareArrays(output, expected));
}

// Test 6: Sigmoid Activation
TEST_F(MLOperationsTest, SigmoidActivation) {
    std::vector<float> input = {-2, -1, 0, 1, 2};
    std::vector<float> output(input.size());
    
    // Apply Sigmoid: 1 / (1 + exp(-x))
    std::transform(input.begin(), input.end(), output.begin(),
                  [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    
    // Check specific values
    EXPECT_NEAR(output[2], 0.5f, 1e-5f);        // sigmoid(0) = 0.5
    EXPECT_NEAR(output[0] + output[4], 1.0f, 1e-5f); // sigmoid(-x) + sigmoid(x) = 1
}

// Test 7: Tanh Activation
TEST_F(MLOperationsTest, TanhActivation) {
    std::vector<float> input = {-2, -1, 0, 1, 2};
    std::vector<float> output(input.size());
    
    // Apply Tanh
    std::transform(input.begin(), input.end(), output.begin(),
                  [](float x) { return std::tanh(x); });
    
    // Check properties
    EXPECT_NEAR(output[2], 0.0f, 1e-5f);        // tanh(0) = 0
    EXPECT_NEAR(output[0], -output[4], 1e-5f);  // tanh(-x) = -tanh(x)
    EXPECT_LT(output[4], 1.0f);                 // tanh(x) < 1
    EXPECT_GT(output[4], 0.0f);                 // tanh(2) > 0
}

// Test 8: Max Pooling 2D
TEST_F(MLOperationsTest, MaxPooling2D) {
    // 4x4 input with 2x2 max pooling, stride 2
    const int inputH = 4, inputW = 4;
    const int poolH = 2, poolW = 2;
    const int strideH = 2, strideW = 2;
    const int outputH = inputH / strideH;
    const int outputW = inputW / strideW;
    
    std::vector<float> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    std::vector<float> output(outputH * outputW);
    
    // Apply max pooling
    for (int oh = 0; oh < outputH; ++oh) {
        for (int ow = 0; ow < outputW; ++ow) {
            float maxVal = -std::numeric_limits<float>::infinity();
            for (int ph = 0; ph < poolH; ++ph) {
                for (int pw = 0; pw < poolW; ++pw) {
                    int ih = oh * strideH + ph;
                    int iw = ow * strideW + pw;
                    if (ih < inputH && iw < inputW) {
                        maxVal = std::max(maxVal, input[ih * inputW + iw]);
                    }
                }
            }
            output[oh * outputW + ow] = maxVal;
        }
    }
    
    std::vector<float> expected = {6, 8, 14, 16};
    EXPECT_TRUE(compareArrays(output, expected));
}

// Test 9: Average Pooling 2D
TEST_F(MLOperationsTest, AveragePooling2D) {
    // 4x4 input with 2x2 average pooling, stride 2
    const int inputH = 4, inputW = 4;
    const int poolH = 2, poolW = 2;
    const int strideH = 2, strideW = 2;
    const int outputH = inputH / strideH;
    const int outputW = inputW / strideW;
    
    std::vector<float> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    std::vector<float> output(outputH * outputW);
    
    // Apply average pooling
    for (int oh = 0; oh < outputH; ++oh) {
        for (int ow = 0; ow < outputW; ++ow) {
            float sum = 0.0f;
            int count = 0;
            for (int ph = 0; ph < poolH; ++ph) {
                for (int pw = 0; pw < poolW; ++pw) {
                    int ih = oh * strideH + ph;
                    int iw = ow * strideW + pw;
                    if (ih < inputH && iw < inputW) {
                        sum += input[ih * inputW + iw];
                        count++;
                    }
                }
            }
            output[oh * outputW + ow] = sum / count;
        }
    }
    
    std::vector<float> expected = {3.5f, 5.5f, 11.5f, 13.5f};
    EXPECT_TRUE(compareArrays(output, expected));
}

// Test 10: Batch Normalization
TEST_F(MLOperationsTest, BatchNormalization) {
    const int batch = 4, channels = 2, height = 2, width = 2;
    const int spatialSize = height * width;
    
    // Input: NCHW format
    std::vector<float> input(batch * channels * spatialSize);
    for (int i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i);
    }
    
    // Compute mean and variance per channel
    std::vector<float> mean(channels, 0.0f);
    std::vector<float> variance(channels, 0.0f);
    
    for (int c = 0; c < channels; ++c) {
        // Compute mean
        float sum = 0.0f;
        int count = 0;
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < spatialSize; ++s) {
                int idx = n * channels * spatialSize + c * spatialSize + s;
                sum += input[idx];
                count++;
            }
        }
        mean[c] = sum / count;
        
        // Compute variance
        float varSum = 0.0f;
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < spatialSize; ++s) {
                int idx = n * channels * spatialSize + c * spatialSize + s;
                float diff = input[idx] - mean[c];
                varSum += diff * diff;
            }
        }
        variance[c] = varSum / count;
    }
    
    // Apply batch normalization
    const float epsilon = 1e-5f;
    std::vector<float> gamma(channels, 1.0f); // Scale
    std::vector<float> beta(channels, 0.0f);  // Shift
    std::vector<float> output(input.size());
    
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int s = 0; s < spatialSize; ++s) {
                int idx = n * channels * spatialSize + c * spatialSize + s;
                float normalized = (input[idx] - mean[c]) / std::sqrt(variance[c] + epsilon);
                output[idx] = gamma[c] * normalized + beta[c];
            }
        }
    }
    
    // Verify mean is close to 0 and variance is close to 1 after normalization
    for (int c = 0; c < channels; ++c) {
        float newMean = 0.0f;
        float newVar = 0.0f;
        int count = 0;
        
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < spatialSize; ++s) {
                int idx = n * channels * spatialSize + c * spatialSize + s;
                newMean += output[idx];
                count++;
            }
        }
        newMean /= count;
        
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < spatialSize; ++s) {
                int idx = n * channels * spatialSize + c * spatialSize + s;
                float diff = output[idx] - newMean;
                newVar += diff * diff;
            }
        }
        newVar /= count;
        
        EXPECT_NEAR(newMean, 0.0f, 1e-4f);
        EXPECT_NEAR(newVar, 1.0f, 1e-2f);
    }
}

// Test 11: Softmax
TEST_F(MLOperationsTest, Softmax) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f};
    std::vector<float> output(input.size());
    
    // Compute softmax
    float maxVal = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    
    // Compute exp(x - max) for numerical stability
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxVal);
        sum += output[i];
    }
    
    // Normalize
    for (float& val : output) {
        val /= sum;
    }
    
    // Verify properties
    float outputSum = std::accumulate(output.begin(), output.end(), 0.0f);
    EXPECT_NEAR(outputSum, 1.0f, 1e-5f); // Sum should be 1
    
    // Verify ordering preserved
    for (size_t i = 1; i < input.size(); ++i) {
        if (input[i] > input[i-1]) {
            EXPECT_GT(output[i], output[i-1]);
        }
    }
}

// Test 12: INT8 Quantization
TEST_F(MLOperationsTest, INT8Quantization) {
    std::vector<float> input = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    
    // Compute scale and zero point
    float minVal = *std::min_element(input.begin(), input.end());
    float maxVal = *std::max_element(input.begin(), input.end());
    
    float scale = (maxVal - minVal) / 255.0f;
    int zeroPoint = static_cast<int>(std::round(-minVal / scale));
    
    // Quantize
    std::vector<int8_t> quantized(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        int q = static_cast<int>(std::round(input[i] / scale + zeroPoint));
        quantized[i] = static_cast<int8_t>(std::max(-128, std::min(127, q - 128)));
    }
    
    // Dequantize
    std::vector<float> dequantized(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        dequantized[i] = scale * (quantized[i] + 128 - zeroPoint);
    }
    
    // Check reconstruction error
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(dequantized[i], input[i], scale * 2); // Within 2 quantization levels
    }
}

// Test 13: Layer Normalization
TEST_F(MLOperationsTest, LayerNormalization) {
    const int batch = 2, seqLen = 3, hidden = 4;
    
    // Input: [batch, seqLen, hidden]
    auto input = generateRandomTensor(batch * seqLen * hidden);
    std::vector<float> output(input.size());
    
    // Normalize over hidden dimension
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seqLen; ++s) {
            // Compute mean and variance for this position
            float mean = 0.0f;
            for (int h = 0; h < hidden; ++h) {
                int idx = b * seqLen * hidden + s * hidden + h;
                mean += input[idx];
            }
            mean /= hidden;
            
            float variance = 0.0f;
            for (int h = 0; h < hidden; ++h) {
                int idx = b * seqLen * hidden + s * hidden + h;
                float diff = input[idx] - mean;
                variance += diff * diff;
            }
            variance /= hidden;
            
            // Normalize
            const float epsilon = 1e-5f;
            for (int h = 0; h < hidden; ++h) {
                int idx = b * seqLen * hidden + s * hidden + h;
                output[idx] = (input[idx] - mean) / std::sqrt(variance + epsilon);
            }
        }
    }
    
    // Verify normalization
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seqLen; ++s) {
            float mean = 0.0f;
            float var = 0.0f;
            
            for (int h = 0; h < hidden; ++h) {
                int idx = b * seqLen * hidden + s * hidden + h;
                mean += output[idx];
            }
            mean /= hidden;
            
            for (int h = 0; h < hidden; ++h) {
                int idx = b * seqLen * hidden + s * hidden + h;
                float diff = output[idx] - mean;
                var += diff * diff;
            }
            var /= hidden;
            
            EXPECT_NEAR(mean, 0.0f, 1e-4f);
            EXPECT_NEAR(var, 1.0f, 1e-2f);
        }
    }
}

// Test 14: Transpose Operation
TEST_F(MLOperationsTest, Transpose) {
    // Transpose 3x4 matrix to 4x3
    const int rows = 3, cols = 4;
    
    std::vector<float> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    
    std::vector<float> output(rows * cols);
    
    // Perform transpose
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            output[c * rows + r] = input[r * cols + c];
        }
    }
    
    // Verify specific elements
    EXPECT_EQ(output[0], 1);  // (0,0) -> (0,0)
    EXPECT_EQ(output[1], 5);  // (1,0) -> (0,1)
    EXPECT_EQ(output[3], 2);  // (0,1) -> (1,0)
    EXPECT_EQ(output[11], 12); // (2,3) -> (3,2)
}

// Test 15: GELU Activation
TEST_F(MLOperationsTest, GELUActivation) {
    std::vector<float> input = {-2, -1, 0, 1, 2};
    std::vector<float> output(input.size());
    
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
    
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_pi * (x + 0.044715f * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    
    // Verify properties
    EXPECT_NEAR(output[2], 0.0f, 1e-5f); // GELU(0) ≈ 0
    EXPECT_GT(output[3], 0.0f);          // GELU(1) > 0
    EXPECT_LT(output[1], 0.0f);          // GELU(-1) < 0
    
    // GELU should be smooth and monotonic
    for (size_t i = 1; i < output.size(); ++i) {
        EXPECT_GT(output[i], output[i-1]);
    }
}

} // namespace mlsdk::tests