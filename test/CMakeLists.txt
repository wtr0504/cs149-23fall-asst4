# cmake_minimum_required(VERSION 3.10)
# project(demo)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ..)
# set(CMAKE_PREFIX_PATH "~/libtorch/include/") # 改成你自己下载的libtorch的地址
# link_directories(/usr/local/cuda-12.2/lib64) # 改成你自己安装cuda的地址
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc) # 同上
# enable_language(CUDA) #启动cuda语言
# find_package(Torch REQUIRED)
# add_executable(demo test.cpp)
# target_link_libraries(demo ${TORCH_LIBRARIES})

cmake_minimum_required(VERSION 3.0)

project(main)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(main test.cpp)
target_compile_features(main PUBLIC cxx_range_for)
target_link_libraries(main ${TORCH_LIBRARIES})
set_property(TARGET main PROPERTY CXX_STANDARD 17)