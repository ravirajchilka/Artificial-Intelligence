cmake_minimum_required(VERSION 3.10)
project(MyProject CUDA CXX)

cmake_policy(SET CMP0146 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA Toolkit root directory
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")

# Find CUDAToolkit instead of CUDA
find_package(CUDAToolkit 11.8 REQUIRED)

# Set Torch root directory
set(TORCH_ROOT "D:/Downloads/libtorch-win-shared-with-deps-debug-2.4.0+cu118/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_ROOT})

# Find Torch
find_package(Torch REQUIRED)

# cuDNN configuration
option(USE_CUDNN "Use cuDNN" ON)
if(USE_CUDNN)
  find_library(CUDNN_LIBRARY cudnn
               HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
  if(CUDNN_LIBRARY)
    message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
    add_definitions(-DUSE_CUDNN)
    include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
    list(APPEND TORCH_LIBRARIES ${CUDNN_LIBRARY})
  else()
    message(WARNING "cuDNN not found. Compiling without cuDNN support.")
    set(USE_CUDNN OFF)
  endif()
endif()

# Your executable
add_executable(MyApp main.cpp)
target_link_libraries(MyApp ${TORCH_LIBRARIES} CUDA::cudart)


if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET MyApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS}
    $<TARGET_FILE_DIR:MyApp>
  )
endif(MSVC)



# After your add_executable() command:
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET MyApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS}
    $<TARGET_FILE_DIR:MyApp>
  )
endif(MSVC)


if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_ROOT}/lib/*.dll")
  add_custom_command(TARGET MyApp
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:MyApp>)
endif(MSVC)

