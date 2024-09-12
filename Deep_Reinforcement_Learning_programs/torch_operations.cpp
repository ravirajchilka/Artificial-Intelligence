#include <torch/torch.h>
#include <iostream>

int main() {

    auto emptyT = torch::empty({3,2}); // 2,3,3 would represent two 3x3 matrices whereas 4,2,2,2 would 
                                       // represent four sets of two 2x2 matrices
    std::cout << emptyT << std::endl;
  

    auto torchRand = torch::rand({3,4}); // random values in 3x4 matrix
    std:: cout << torchRand << std::endl;                                        


    auto torchOnes = torch::ones({3,4}); // 1 values in 3x4 matrix
    std:: cout << torchOnes << std::endl;                                        


    auto tensorDouble = torch::ones({4,3},torch::kDouble); // default dtype is float, 
                                                    // use torch::kDouble to convert it to Double or Int 
    std::cout << tensorDouble.dtype() << std::endl;
    std::cout << "size" << tensorDouble.size(0) << std::endl; // size of specefic dimension
    std::cout << "sizes" << tensorDouble.sizes() << std::endl; // use sizes for size of all dimension


    auto customTensor = torch::tensor({5.3,9.0}); // create custom tensor
    std::cout << "customTensos" << customTensor << std::endl;


}



