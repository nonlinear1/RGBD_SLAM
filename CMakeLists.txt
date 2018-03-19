cmake_minimum_required(VERSION 3.9)
project(daniel_slam)

set(CMAKE_CXX_STANDARD 11)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

find_package(PCL 1.8 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sophus REQUIRED)
find_package(G2O REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)

set(Containers_INCLUDE containers/include)
set(Cuda_INCLUDE cuda/include)
set(Utils_INCLUDE utils)
set(Base_INCLUDE include)

include_directories(
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Containers_INCLUDE}
        ${Utils_INCLUDE}
        ${Base_INCLUDE}
        ${Cuda_INCLUDE}
        ${SOPHUS_HEADER_FILES}
        ${G2O_INCLUDE_DIR}
)

add_definitions(${PCL_DEFINITIONS})

file(GLOB containers containers/src/*.cpp)
file(GLOB src_file src/*.cpp)
file(GLOB cuda cuda/*.cu)

set(SOURCE_FILES main.cpp)
set(ICP_TEST_FILES icp_test.cpp)
set(ICP_SEQ_FILES icp_seq_test.cpp)
set(Cuda_TEST cuda_test.cu)

CUDA_COMPILE(cuda_objs ${cuda})

add_executable(daniel_slam
        ${SOURCE_FILES}
        ${containers}
        ${cuda_objs}
        ${src_file})

add_executable(g2o_test
        g2o_test.cpp)

add_executable(ICPtest
        ${ICP_TEST_FILES}
        ${containers}
        ${cuda_objs}
        src/rgbicp.cpp)

add_executable(VOtest
        ${ICP_SEQ_FILES}
        ${containers}
        ${cuda_objs}
        src/rgbicp.cpp)

cuda_add_executable(cuda_test
        ${Cuda_TEST})

target_link_libraries(daniel_slam
        ${PCL_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        ${SophusTargets}
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${G2O_SOLVER_DENSE}
        ${G2O_SOLVER_EIGEN}
        ${G2O_TYPES_SLAM3D})

target_link_libraries(g2o_test
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${G2O_SOLVER_DENSE}
        ${G2O_SOLVER_EIGEN}
        ${G2O_TYPES_SLAM3D})

target_link_libraries(ICPtest
        ${PCL_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        ${SophusTargets})

target_link_libraries(VOtest
        ${PCL_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        ${SophusTargets})
