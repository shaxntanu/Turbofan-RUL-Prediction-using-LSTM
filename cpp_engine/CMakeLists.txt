cmake_minimum_required(VERSION 3.16)
project(turbofan_cpp_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TURBOFAN_CPP_OPENMP "Enable OpenMP when available" ON)
option(TURBOFAN_CPP_BUILD_MODULE "Build the optional pybind11 module" OFF)
option(TURBOFAN_CPP_BUILD_BENCH "Build the standalone benchmark" ON)

add_library(turbofan_engine STATIC src/pca.cpp src/kmeans.cpp)
target_include_directories(turbofan_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(TURBOFAN_CPP_OPENMP)
  find_package(OpenMP COMPONENTS CXX QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(turbofan_engine PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(TURBOFAN_CPP_BUILD_BENCH)
  add_executable(turbofan_cpp_bench benchmarks/bench_main.cpp)
  target_link_libraries(turbofan_cpp_bench PRIVATE turbofan_engine)
endif()

if(TURBOFAN_CPP_BUILD_MODULE)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT)
  if(PYBIND11_RESULT EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  endif()
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(turbofan_cpp src/bindings.cpp)
  target_link_libraries(turbofan_cpp PRIVATE turbofan_engine)
endif()
