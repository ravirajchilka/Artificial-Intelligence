#include <torch/torch.h>
#include <iostream>

int main() {
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available" << std::endl;
        
        // Check if cuDNN is available
        if (torch::cuda::cudnn_is_available()) {
            std::cout << "cuDNN is available" << std::endl;
        } else {
            std::cout << "cuDNN is not available" << std::endl;
        }
    } else {
        std::cout << "CUDA is not available" << std::endl;
    }

    return 0;
}
