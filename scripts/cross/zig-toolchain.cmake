# Generic CMake toolchain for cross-compiling ncnn with the zig shims.
#
# Usage:
#   cmake -S thirdparty/ncnn -B <build> \
#     -DCMAKE_TOOLCHAIN_FILE=scripts/cross/zig-toolchain.cmake \
#     -DZIG_SHIM_DIR=<dir from make_shims.sh> -DZIG_SYSTEM_NAME=Linux|Windows \
#     <ncnn options...>
set(CMAKE_SYSTEM_NAME "${ZIG_SYSTEM_NAME}")
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER   "${ZIG_SHIM_DIR}/cc")
set(CMAKE_CXX_COMPILER "${ZIG_SHIM_DIR}/c++")
set(CMAKE_AR           "${ZIG_SHIM_DIR}/ar")
set(CMAKE_RANLIB       "${ZIG_SHIM_DIR}/ranlib")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
