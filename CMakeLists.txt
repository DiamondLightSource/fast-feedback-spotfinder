cmake_minimum_required(VERSION 3.20...3.30)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+

# Get version at start so that we can set it at root level
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
include(ResolveGitVersion)

project(fast-feedback-service LANGUAGES NONE VERSION ${FFS_VERSION_CMAKE})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

include_directories(include)

# Dependency fetching
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.0.2
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(fmt)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nlohmann_json)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY  https://github.com/ndevenish/argparse
    GIT_TAG f362c4647e7b4bbfef8320040409560b5f90e9e0
)
FetchContent_MakeAvailable(argparse)
if(dev) #flag for for development
    add_subdirectory(dx2)
else()
    FetchContent_Declare(
        dx2
        GIT_REPOSITORY  https://github.com/dials/dx2
        GIT_TAG 2654929bc73e0d699dfdfcc2d924c3454e434ccf
    )
    FetchContent_MakeAvailable(dx2)
endif()

# Make a small library that we can link to to get the version
project(version CXX)
configure_file(version.cc.in version.cc @ONLY)
add_library(version STATIC version.cc)

add_subdirectory(h5read)
add_subdirectory(baseline)
add_subdirectory(spotfinder)
add_subdirectory(baseline_indexer)
