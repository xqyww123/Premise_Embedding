cmake_minimum_required(VERSION 3.10)
project(isabelle_vector VERSION 1.0 LANGUAGES C CXX)

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

    # NO_DEFAULT_PATH matters: without it, find_path searches the system prefixes
    # *before* PATHS, so /usr/include wins and the headers come from whatever
    # libhwy-dev the distro ships (1.0.7 here) while the static library above is
    # the vendored 1.3.0. Highway's ops are header-only, but libhwy.a carries the
    # runtime target detection and the dispatch tables, whose target numbering is
    # not stable across releases -- a mismatch can dispatch into the wrong entry.
    find_path(HWY_INCLUDE_DIR hwy/highway.h
        NO_DEFAULT_PATH
        PATHS
            ${CMAKE_SOURCE_DIR}/contrib/highway-1.3.0
            ${CMAKE_SOURCE_DIR}/contrib/highway
            /usr/local/include
            /usr/include
    )

    message(STATUS "DEBUG: HWY_STATIC_LIBRARY = ${HWY_STATIC_LIBRARY}")
    message(STATUS "DEBUG: HWY_INCLUDE_DIR = ${HWY_INCLUDE_DIR}")

    if(HWY_STATIC_LIBRARY AND HWY_INCLUDE_DIR)
        # Headers and library must come from the same source tree.
        get_filename_component(_hwy_lib_root "${HWY_STATIC_LIBRARY}" DIRECTORY)
        get_filename_component(_hwy_lib_root "${_hwy_lib_root}" DIRECTORY)
        if(NOT "${HWY_INCLUDE_DIR}" STREQUAL "${_hwy_lib_root}")
            message(FATAL_ERROR
                "Highway headers and static library come from different trees:\n"
                "  headers: ${HWY_INCLUDE_DIR}\n  library: ${HWY_STATIC_LIBRARY}\n"
                "Their versions must match; libhwy.a owns the dispatch tables.")
        endif()
        file(STRINGS "${HWY_INCLUDE_DIR}/hwy/base.h" _hwy_ver
             REGEX "^#define HWY_(MAJOR|MINOR|PATCH) ")
        string(REGEX MATCHALL "[0-9]+" _hwy_ver "${_hwy_ver}")
        string(REPLACE ";" "." _hwy_ver "${_hwy_ver}")
        message(STATUS "Highway ${_hwy_ver} (headers and static library from ${HWY_INCLUDE_DIR})")
        set(HWY_FOUND TRUE)
    else()
        message(FATAL_ERROR "Highway static library not found in ./contrib!\nStatic lib: ${HWY_STATIC_LIBRARY}\nInclude dir: ${HWY_INCLUDE_DIR}")
    endif()
endif()

# This library contains no Python. The gather glue that used to live here is now
# Isabelle_Semantic_Embedding/_vecgather.c, a CPython extension module built by
# setuptools. Keeping the kernel Python-free is what lets Isabelle/ML load this
# very file into a process with no interpreter -- on Unix through dlopen(RTLD_LAZY)
# and on Windows through LoadLibrary, which resolves the whole import table eagerly
# and would therefore not tolerate a python3.dll import at all.

# A fat binary covers both Intel Macs (x86_64-darwin) and Apple Silicon
# (arm64-darwin, which Isabelle prefers by default: etc/options ML_system_apple).
# This MUST precede add_library: a target's OSX_ARCHITECTURES property is
# initialized from the variable at target-creation time, and setting it afterwards
# silently yields a host-architecture-only library.
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "universal2")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "matches the macosx_11_0 wheel tag")
endif()

add_library(isabelle_vector SHARED isabelle_vector.cpp)

# Turn a stray undefined symbol into a link error. Removing the old
# --allow-shlib-undefined would NOT have done this: ld permits undefined references
# from a -shared link's own objects by default, and that option only ever governed
# symbols reached through DT_NEEDED libraries. So this is the build-time guard that
# catches a CPython symbol creeping back into the kernel; test_ml_dlopen is the
# run-time one. (macOS errors on undefined symbols by default; PE cannot have any.)
if(NOT WIN32 AND NOT APPLE)
    target_link_options(isabelle_vector PRIVATE "LINKER:-z,defs")
endif()

# MinGW-w64, never MSVC: Highway marks AVX3 and everything above it as
# HWY_BROKEN_MSVC (detect_targets.h:191), silently dropping five SIMD targets.
# Poly/ML is itself built with MinGW (Admin/component_polyml.scala:37).
if(WIN32)
    set_target_properties(isabelle_vector PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

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

# SIMD dot + zero-copy gather tests (test_topk.cpp only covers the scalar heap)
add_executable(test_gather test_gather.cpp)
target_link_libraries(test_gather PRIVATE isabelle_vector m)
target_compile_options(test_gather PRIVATE -O2)

# Reproduces Poly/ML's dlopen(RTLD_LAZY) so a stray Python data symbol fails the
# build here rather than at Isabelle load time.
add_executable(test_ml_dlopen test_ml_dlopen.c)
target_link_libraries(test_ml_dlopen PRIVATE ${CMAKE_DL_LIBS})

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
