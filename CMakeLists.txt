cmake_minimum_required(VERSION 3.10)
project(BetaZero)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译优化选项
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_BUILD_TYPE Release)

# LibTorch设置
set(Torch_DIR /home/liznb/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

# 包含目录
include_directories(include ${TORCH_INCLUDE_DIRS})

# 查找源文件
file(GLOB_RECURSE COMMON_SOURCES "src/*.cpp")
list(FILTER COMMON_SOURCES EXCLUDE REGEX "src/main.cpp|src/test.cpp")

# 创建可执行文件
add_executable(${PROJECT_NAME} src/main.cpp ${COMMON_SOURCES})
add_executable(test src/test.cpp ${COMMON_SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(test PRIVATE ${TORCH_LIBRARIES})

# 文件系统支持
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
    target_link_libraries(test PRIVATE stdc++fs)
endif()

# 设置输出目录
set_target_properties(${PROJECT_NAME} test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")