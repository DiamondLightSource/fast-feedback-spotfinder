project(indexer CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(spdlog)
find_package (Threads)

# Automatic Dependencies
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft
    GIT_TAG cpp
)
FetchContent_MakeAvailable(Eigen3)
FetchContent_MakeAvailable(pocketfft)

FetchContent_GetProperties(pocketfft)
add_library(pocketfft INTERFACE)
target_include_directories(pocketfft INTERFACE ${pocketfft_SOURCE_DIR})

add_executable(baseline_indexer
    indexer.cc
)
target_link_libraries(baseline_indexer
    PRIVATE
    fmt
    dx2
    Eigen3::Eigen
    pocketfft
    argparse
    nlohmann_json::nlohmann_json
    ${CMAKE_THREAD_LIBS_INIT}
)
