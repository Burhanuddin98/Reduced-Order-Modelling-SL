cmake_minimum_required(VERSION 3.18)
project(helmholtz LANGUAGES CUDA CXX)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Build shared library from CUDA source
add_library(helmholtz SHARED helmholtz_gpu.cu)

# Link cuSOLVER and cuSPARSE
target_link_libraries(helmholtz
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cudart
)

# Set properties
set_target_properties(helmholtz PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "75"  # RTX 2060 = SM 7.5
)

# Windows: export all symbols
if(WIN32)
    set_target_properties(helmholtz PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# Install
install(TARGETS helmholtz DESTINATION ${CMAKE_SOURCE_DIR})
