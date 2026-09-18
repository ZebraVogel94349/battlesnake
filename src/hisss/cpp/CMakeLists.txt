cmake_minimum_required(VERSION 3.15)
project(hisss_link CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
option(
    HISSS_REQUIRE_CUDA
    "Fail configuration instead of building the CUDA stub when nvcc is unavailable"
    OFF
)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18" AND NOT CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.1")
            set(HISSS_DEFAULT_CUDA_ARCHITECTURES 80 86)
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
                list(APPEND HISSS_DEFAULT_CUDA_ARCHITECTURES 89 90)
            endif()
            # CUDA 12.8 added native Blackwell targets. In particular, sm_120
            # prevents RTX 5090 hosts from falling back to JIT-compiled PTX
            # emitted for the much older sm_86 target.
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
                list(APPEND HISSS_DEFAULT_CUDA_ARCHITECTURES 100 120)
            endif()
            set(
                CMAKE_CUDA_ARCHITECTURES
                ${HISSS_DEFAULT_CUDA_ARCHITECTURES}
            )
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
    endif()
    set(CUDA_BACKEND_SOURCE source/battlesnake_cuda.cu)
    message(
        STATUS
        "Building hisss CUDA backend with ${CMAKE_CUDA_COMPILER}; "
        "architectures=${CMAKE_CUDA_ARCHITECTURES}"
    )
else()
    if(HISSS_REQUIRE_CUDA)
        message(
            FATAL_ERROR
            "HISSS_REQUIRE_CUDA=ON, but CMake could not find nvcc. "
            "Install the CUDA toolkit and/or set CUDACXX=/path/to/nvcc."
        )
    endif()
    set(CUDA_BACKEND_SOURCE source/battlesnake_cuda_stub.cpp)
    message(STATUS "CUDA compiler not found; building hisss CUDA stub")
endif()

if(MSVC)
    add_compile_options(/permissive-)
endif()

set(ALGLIB_SOURCES
    alglib/ap.cpp
    alglib/alglibinternal.cpp
    alglib/alglibmisc.cpp
    alglib/linalg.cpp
    alglib/solvers.cpp
    alglib/optimization.cpp
)

set(PROJECT_SOURCES
    source/battlesnake.cpp
    source/battlesnake_helper.cpp
    source/utils.cpp
    source/nash.cpp
    source/link.cpp
)

add_library(link SHARED ${PROJECT_SOURCES} ${CUDA_BACKEND_SOURCE} ${ALGLIB_SOURCES})

target_include_directories(link PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/header
    ${CMAKE_CURRENT_SOURCE_DIR}/alglib
)

set_target_properties(link PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -O3 for GCC/Clang, /O2 for MSVC
target_compile_options(link PRIVATE
    $<$<BOOL:${MSVC}>:/O2>
    $<$<NOT:$<BOOL:${MSVC}>>:-O3>
)

# Alglib: suppress warnings (mirrors hatch_build.py -w)
foreach(src IN LISTS ALGLIB_SOURCES)
    set_source_files_properties(${src} PROPERTIES
        COMPILE_FLAGS "$<IF:$<BOOL:${MSVC}>,/w,-w>")
endforeach()

# Project sources: enable -Wall (mirrors hatch_build.py -Wall)
foreach(src IN LISTS PROJECT_SOURCES)
    set_source_files_properties(${src} PROPERTIES
        COMPILE_FLAGS "$<IF:$<BOOL:${MSVC}>,/W3,-Wall>")
endforeach()

# Install to hisss/cpp/ so lib.py finds it at the expected path
# LIBRARY = .so/.dylib on Linux/macOS; RUNTIME = .dll on Windows
install(TARGETS link
    LIBRARY DESTINATION hisss/cpp
    RUNTIME DESTINATION hisss/cpp
)
