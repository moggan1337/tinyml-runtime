cmake_minimum_required(VERSION 3.20)
project(tinyml-runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# EMBEDDED: strict subset — no RTTI, no exceptions, no STL streams
add_compile_options(
  -Wall -Wextra -Wpedantic -Wconversion
  -fno-rtti -fno-exceptions
  -ffunction-sections -fdata-sections
)

# Optional CMSIS-NN backend
option(TINYML_USE_CMSIS_NN "Enable CMSIS-NN kernel backend" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerOptions)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

add_library(tinyml STATIC
  src/tinyml/version.cpp
  src/tinyml/model.cpp
)
target_link_libraries(tinyml PUBLIC)
apply_warnings_as_errors(tinyml)

# Minimal test framework — each test_xxx returns 0 on pass, 1 on fail
# Glob all test source files
file(GLOB TINYML_TEST_SOURCES CONFIGURE_DEPENDS "tests/unit/*.cpp")
add_executable(tinyml_test ${TINYML_TEST_SOURCES})
target_link_libraries(tinyml_test PRIVATE tinyml)
apply_warnings_as_errors(tinyml_test)
include_directories(tinyml_test PRIVATE tests/unit)

# Test runner target
add_custom_target(test
  COMMAND tinyml_test
  DEPENDS tinyml_test
  COMMENT "Run all unit tests")

# Cross-compile target (ARM Cortex-M4/M7)
if(CMAKE_CROSSCOMPILING)
  find_program(ARM_TOOLCHAIN arm-none-eabi-g++ REQUIRED)
  set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
  set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
  set(CMAKE_EXE_LINKER_FLAG "-specs=nosys.specs -T${CMAKE_SOURCE_DIR}/linker/stm32f4.ld")
endif()

install(TARGETS tinyml ARCHIVE DESTINATION lib)
install(DIRECTORY include/tinyml DESTINATION include)
