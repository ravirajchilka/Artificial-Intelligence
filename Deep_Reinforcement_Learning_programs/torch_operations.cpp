#include <torch/torch.h>
#include <iostream>

int main() {
    // 2,3,3 would represent two 3x3 matrices whereas 4,2,2,2 would 
    auto emptyT = torch::empty({3,2}); // represent four sets of two 2x2 matrices
                                       
    std::cout << emptyT << std::endl;
  
    // random values in 3x4 matrix
    auto torchRand = torch::rand({3,4}); 
    std:: cout << torchRand << std::endl;                                        


    // 1 values in 3x4 matrix
    auto torchOnes = torch::ones({3,4}); 
    std:: cout << torchOnes << std::endl;                                        


    // default dtype is float, 
    auto tensorDouble = torch::ones({4,3},torch::kDouble); // use torch::kDouble to convert it to Double or Int 
                                                    
    std::cout << tensorDouble.dtype() << std::endl;
    std::cout << "size" << tensorDouble.size(0) << std::endl; // size of specefic dimension
    std::cout << "sizes" << tensorDouble.sizes() << std::endl; // use sizes for size of all dimension


    // create custom tensor
    auto customTensor = torch::tensor({5.3,9.0}); 
    std::cout << "customTensos" << customTensor << std::endl;


   // addition, sub, mult, div
    auto x = torch::rand({2,2});
    auto y = torch::rand({2,2});

    std::cout << x << std::endl;
    std::cout << y << std::endl;

    y.add_(x); // in place addition
    auto z = torch::add(x,y); // added to z similar to auto z = x+y;
    std::cout << y;

}



