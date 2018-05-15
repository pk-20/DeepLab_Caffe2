# requires a minimum of 2.8.12.
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  find_package(Caffe2 REQUIRED)
endif()


set(CAFFE2_PATH /home/Caffe2/caffe2/)
set(CUDA_INCLUDES_PATH /home/cuda/include/)

set(OPS_LIB caffe2_CUST_OPS_AddRemovePadding)
set(OPS_1 ${CMAKE_CURRENT_SOURCE_DIR}/src/add_remove_padding_op)


list(APPEND CUDA_INCLUDE_DIRS -I${CAFFE2_INCLUDE_DIRS})


set(CUDA_NVCC_FLAGS -std=c++11)


cuda_add_library(${OPS_LIB} SHARED
    ${OPS_1}.cc
    #${OPS_1}.cu
    )

target_include_directories(${OPS_LIB}
    PRIVATE
    /usr/include/eigen3
    ${CAFFE2_PATH}
    ${CUDA_INCLUDES_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/)

target_link_libraries(${OPS_LIB} caffe2)
install(TARGETS ${OPS_LIB} DESTINATION lib)
