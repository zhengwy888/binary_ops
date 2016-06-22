cmake_minimum_required(VERSION 2.8)

project(xnor_lstm C CXX)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2 -D GOOGLE_CUDA=1")        ## Optimize

# CUDA flags
find_package(CUDA QUIET REQUIRED COMPONENTS cuda cudart)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -D GOOGLE_CUDA=1 -x cu -Xcompiler" )

execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include())" 2>/dev/null
    OUTPUT_VARIABLE TF_INCLUDE)

MESSAGE(STATUS "Found Tensorflow headers: " ${TF_INCLUDE})
include_directories(${TF_INCLUDE})

#include
cuda_add_library(bnmatmul_op_gpu SHARED
    bnmatmul_op_gpu.cu)
CUDA_ADD_CUBLAS_TO_TARGET(bnmatmul_op_gpu)
target_link_libraries(bnmatmul_op_gpu
    ${CUDA_LIBRARIES})

add_library(bnmatmul_op SHARED
    bnmatmul_op.cc)
SET_TARGET_PROPERTIES(bnmatmul_op PROPERTIES PREFIX "")
target_link_libraries(bnmatmul_op 
    bnmatmul_op_gpu
    )
