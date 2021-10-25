cmake_minimum_required(VERSION 3.15)
project(eoneural)

# C libraries

add_subdirectory(src/eoneural)
add_subdirectory(src/bin/eoneural)

# C++ libraries

find_package(GTest REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

include(FetchContent)

FetchContent_Declare(
        rapidcsv
        GIT_REPOSITORY https://github.com/d99kris/rapidcsv
        GIT_TAG v8.53
)
FetchContent_MakeAvailable(rapidcsv)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src/eoneural/hpp)
add_subdirectory(src/bin/classification)
add_subdirectory(src/bin/regression)
add_subdirectory(src/test)
