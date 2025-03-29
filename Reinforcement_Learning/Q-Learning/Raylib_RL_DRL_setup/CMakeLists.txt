cmake_minimum_required(VERSION 3.10)

project(RaylibPractice CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Toolkit configuration
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
find_package(CUDAToolkit 12.8 REQUIRED)

# Torch configuration
set(TORCH_ROOT "E:/Programs/libtorch_release_12p6_version_2p6p0/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_ROOT})
find_package(Torch REQUIRED)

# Use cuDNN if available
option(USE_CUDNN "Use cuDNN" ON)
if(USE_CUDNN)
  find_library(CUDNN_LIBRARY cudnn HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
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

# Raylib Configuration
set(RAYLIB_INCLUDE_DIR "E:/Programs/Raylib/raylib/src")
set(RAYLIB_LIB_DIR "E:/Programs/Raylib/raylib/projects/VS2022/build/raylib/bin/x64/Debug")
include_directories(${RAYLIB_INCLUDE_DIR})
link_directories(${RAYLIB_LIB_DIR})

# Source files
set(SOURCES src/main.cpp)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link Raylib and WinMM
if(WIN32)
  target_link_libraries(${PROJECT_NAME} raylib.lib winmm.lib)
else()
  target_link_libraries(${PROJECT_NAME} raylib)
endif()


# Link Torch and CUDA
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} CUDA::cudart)

# MSVC-specific configuration
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_ROOT}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()

# Output Directory Configuration
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
