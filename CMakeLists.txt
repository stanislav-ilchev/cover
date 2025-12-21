cmake_minimum_required(VERSION 3.18)
project(cover_accelerated LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)

# Options
option(BUILD_CUDA "Build CUDA version (requires CUDA toolkit)" OFF)
option(BUILD_OPENMP "Build OpenMP parallel version" ON)

# Optimization flags
if(MSVC)
    set(CMAKE_C_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
else()
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

# =====================================
# OpenMP Parallel Version
# =====================================
if(BUILD_OPENMP)
    find_package(OpenMP)
    
    add_executable(cover_parallel cover_parallel.c)
    
    if(OpenMP_C_FOUND)
        target_link_libraries(cover_parallel PRIVATE OpenMP::OpenMP_C)
        message(STATUS "OpenMP found - building multi-threaded version")
    else()
        message(WARNING "OpenMP not found - building single-threaded version")
    endif()
    
    if(NOT MSVC)
        target_link_libraries(cover_parallel PRIVATE m)
    endif()
endif()

# =====================================
# CUDA Version
# =====================================
if(BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        
        # Detect GPU architecture
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
        endif()
        
        add_executable(cover_cuda cover_cuda.cu)
        
        set_target_properties(cover_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        
        target_compile_options(cover_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
        
        message(STATUS "CUDA found - building GPU-accelerated version")
    else()
        message(WARNING "CUDA not found - skipping GPU version")
    endif()
endif()

# =====================================
# Original cover program (with OpenMP added)
# =====================================
add_executable(cover_original
    cover.c
    bincoef.c
    tables.c
    setoper.c
    anneal.c
    solcheck.c
    exp.c
    arg.c
)

if(OpenMP_C_FOUND)
    target_link_libraries(cover_original PRIVATE OpenMP::OpenMP_C)
endif()

if(NOT MSVC)
    target_link_libraries(cover_original PRIVATE m)
endif()

# Installation
install(TARGETS cover_parallel DESTINATION bin)
if(BUILD_CUDA AND CMAKE_CUDA_COMPILER)
    install(TARGETS cover_cuda DESTINATION bin)
endif()

# Print summary
message(STATUS "")
message(STATUS "Build configuration:")
message(STATUS "  OpenMP parallel version: ${BUILD_OPENMP}")
message(STATUS "  CUDA GPU version: ${BUILD_CUDA}")
message(STATUS "")



