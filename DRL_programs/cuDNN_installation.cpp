#include <torch/torch.h>
#include <iostream>
#include <chrono>

// Helper function for timing
template<typename F>
double time_function(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main() {
    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available" << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
        
        if (torch::cuda::cudnn_is_available()) {
            std::cout << "cuDNN is available" << std::endl;
        } else {
            std::cout << "cuDNN is not available" << std::endl;
        }
    } else {
        std::cout << "CUDA is not available" << std::endl;
        return 0;
    }

    // Test basic tensor operations on CPU and GPU
    const int size = 5000;
    
    // CPU tensor operations
    auto cpu_time = time_function([&]() {
        auto a = torch::rand({size, size});
        auto b = torch::rand({size, size});
        auto c = torch::matmul(a, b);
    });
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;

    // GPU tensor operations
    auto gpu_time = time_function([&]() {
        auto a = torch::rand({size, size}, torch::kCUDA);
        auto b = torch::rand({size, size}, torch::kCUDA);
        auto c = torch::matmul(a, b);
        torch::cuda::synchronize();
    });
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;

    // Test cuDNN-accelerated operation (convolution)
    const int batch_size = 32;
    const int in_channels = 3;
    const int out_channels = 64;
    const int height = 256;
    const int width = 256;

    auto input = torch::rand({batch_size, in_channels, height, width}, torch::kCUDA);
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1));
    conv->to(torch::kCUDA);

    auto conv_time = time_function([&]() {
        auto output = conv->forward(input);
        torch::cuda::synchronize();
    });
    std::cout << "Convolution time (likely using cuDNN): " << conv_time << " ms" << std::endl;

    return 0;
}
