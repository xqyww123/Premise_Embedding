cmake_minimum_required(VERSION 3.10)
project(isabelle_vector VERSION 1.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置构建类型为 Release（如果未指定）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 查找 Highway 库 - 强制使用 contrib 下的静态库
# find_package(HWY QUIET)  # 注释掉，不使用系统的

if(TRUE)  # 强制手动查找
    # 优先查找 ./contrib 下的静态库
    find_library(HWY_STATIC_LIBRARY
        NAMES libhwy.a
        PATHS
            ${CMAKE_SOURCE_DIR}/contrib/highway-1.3.0/build
            ${CMAKE_SOURCE_DIR}/contrib/highway/build
            ${CMAKE_SOURCE_DIR}/contrib/build
        NO_DEFAULT_PATH
    )

    find_path(HWY_INCLUDE_DIR hwy/highway.h
        PATHS
            ${CMAKE_SOURCE_DIR}/contrib/highway-1.3.0
            ${CMAKE_SOURCE_DIR}/contrib/highway
            /usr/local/include
            /usr/include
    )

    message(STATUS "DEBUG: HWY_STATIC_LIBRARY = ${HWY_STATIC_LIBRARY}")
    message(STATUS "DEBUG: HWY_INCLUDE_DIR = ${HWY_INCLUDE_DIR}")

    if(HWY_STATIC_LIBRARY AND HWY_INCLUDE_DIR)
        message(STATUS "Found Highway static library: ${HWY_STATIC_LIBRARY}")
        set(HWY_FOUND TRUE)
    else()
        message(FATAL_ERROR "Highway static library not found in ./contrib!\nStatic lib: ${HWY_STATIC_LIBRARY}\nInclude dir: ${HWY_INCLUDE_DIR}")
    endif()
endif()

# 创建共享库
add_library(isabelle_vector SHARED
    isabelle_vector.cpp
)

# 设置包含目录
target_include_directories(isabelle_vector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # 添加源目录，让 Highway 找到 -inl.h
)

if(HWY_INCLUDE_DIR)
    target_include_directories(isabelle_vector PRIVATE ${HWY_INCLUDE_DIR})
endif()

# 链接 Highway 库 - 使用静态库
if(HWY_STATIC_LIBRARY)
    message(STATUS "Linking static Highway library: ${HWY_STATIC_LIBRARY}")
    target_link_libraries(isabelle_vector PRIVATE ${HWY_STATIC_LIBRARY})
elseif(TARGET hwy)
    target_link_libraries(isabelle_vector PRIVATE hwy)
else()
    message(FATAL_ERROR "Highway library not found!")
endif()

# 设置编译选项 - 关键：启用 Highway 的动态分发
target_compile_options(isabelle_vector PRIVATE
    -O3
    -fPIC
)

# 不使用 -march=native，让 Highway 在运行时选择最佳指令集
# 如果想要针对特定目标优化，可以添加：
# target_compile_definitions(isabelle_vector PRIVATE
#     HWY_COMPILE_ALL_ATTAINABLE=1
# )

# 启用所有可用的 SIMD 目标（推荐）
target_compile_definitions(isabelle_vector PRIVATE
    HWY_COMPILE_ALL_ATTAINABLE=1
)

# 如果只想编译特定目标，可以使用：
# target_compile_definitions(isabelle_vector PRIVATE
#     HWY_DISABLED_TARGETS=HWY_SSE2|HWY_SSSE3
# )

# 设置安装规则（可选）
install(TARGETS isabelle_vector
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 显示配置信息
message(STATUS "==============================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Highway include: ${HWY_INCLUDE_DIR}")
if(HWY_STATIC_LIBRARY)
    message(STATUS "Highway library: ${HWY_STATIC_LIBRARY} (STATIC)")
else()
    message(STATUS "Highway library: ${HWY_LIBRARY} (DYNAMIC)")
endif()
message(STATUS "Dynamic dispatch: ENABLED")
message(STATUS "Multi-target compilation: ENABLED")
message(STATUS "==============================================")
