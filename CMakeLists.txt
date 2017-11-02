cmake_minimum_required(VERSION 2.8.12) # for rpath handling

project(cudaSift)

########################################################################################
# Library setup.                                                                       #
########################################################################################
set(cudaSift_VERSION_MAJOR 3)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(FATAL_ERROR "CUDA not found. ${PROJECT_NAME} cannot not be built.")
endif(NOT CUDA_FOUND)

# Set a default build configuration to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()


########################################################################################
# Options for build configuration.                                                     #
########################################################################################

# Default to shared libraries, override if needbe
option(CUDA_SIFT_BUILD_SHARED "Build a shared library?" ON)
if (CUDA_SIFT_BUILD_SHARED)
  set(CUDA_SIFT_LIBRARY_TYPE "SHARED")
else()
  set(CUDA_SIFT_LIBRARY_TYPE "STATIC")
endif()

# Used to configure defaults depending on whether this project is being built by a
# parent project (if so, do not build the executable or install targets or verbose)
get_directory_property(CUDA_SIFT_HAS_PARENT PARENT_DIRECTORY)
if (CUDA_SIFT_HAS_PARENT)
  set(CUDA_SIFT_VERBOSE_INTERNAL OFF)
else()
  set(CUDA_SIFT_VERBOSE_INTERNAL ON)
endif()

# Allow parent projects to specify whether or not VERBOSE printing is compiled
option(
  CUDA_SIFT_VERBOSE
  "Compile ${PROJECT_NAME} using -DVERBOSE (prints statistics)?"
  ${CUDA_SIFT_VERBOSE_INTERNAL}
)

# Accumulate a list of valid architectures to build, default is all but users can
# override this via the command-line or in a parent project.  Support cycle begins at
# CUDA 7, skipping sm_20 (specify manually if needed, is deprecated and is officially
# unsupported in CUDA 9+.
#
# Thanks to Arnon for keeping this list up to date:
# http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CUDA_SIFT_BUILD_ARCH_INTERNAL "30;35;37;50;52;53")
if (CUDA_VERSION VERSION_GREATER 7.99.99)
  # CUDA 8 adds support for these architectures.
  list(APPEND CUDA_SIFT_BUILD_ARCH_INTERNAL "60;61;62")
endif()
if (CUDA_VERSION VERSION_GREATER 8.99.99)
  # CUDA 9 adds support for the beastly sm_70
  list(APPEND CUDA_SIFT_BUILD_ARCH_INTERNAL "70")
endif()

# Setting this as CACHE STRING is what exposes it to the GUI / parent projects to
# be able to override.
#
# Parent project example override to build just sm_30 and sm_35:
#
#     set(CUDA_SIFT_BUILD_ARCH "30;35" CACHE STRING FORCE)
#
# Example command line usage to build sm_30 and sm_35:
#
#     mkdir build
#     cd build
#     cmake .. -DCUDA_SIFT_BUILD_ARCH="30;35"
set(
  CUDA_SIFT_BUILD_ARCH
  "${CUDA_SIFT_BUILD_ARCH_INTERNAL}"
  CACHE STRING
  "The list (semicolon delineated) of CUDA architectures to build.  If only one GPU, set to e.g. '30'"
)

########################################################################################
# Add the library.                                                                     #
########################################################################################
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

set(lib_headers
  "${CMAKE_CURRENT_SOURCE_DIR}/include/cudaSift/image.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/cudaSift/sift.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/cudaSift/sift_host.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/cudaSift/sift_device.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/cudaSift/utils.h"
)

set(lib_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/src/image.cu"
  # Developer note: sift_device.cu is included directly into sift_host, do not add here
  "${CMAKE_CURRENT_SOURCE_DIR}/src/sift_host.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/matching.cu"
)

if (CUDA_SIFT_VERBOSE)
  if(WIN32)
    add_definitions("/DVERBOSE")
  else()
    add_definitions("-DVERBOSE")
  endif()
endif()

# Disable some Windows warnings.
if (WIN32)
  add_definitions("/D_CRT_SECURE_NO_WARNINGS")
endif()

# Add some optimizations
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2")
if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2") # not for windows?
  if (NOT APPLE)
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo") # Only for linux?
  endif()
endif()

# Build out all of the architecture compilation flags
foreach (arch ${CUDA_SIFT_BUILD_ARCH})
  # virtually real
  list(APPEND CUDA_NVCC_FLAGS "--generate-code arch=compute_${arch},code=sm_${arch}")
  # virtually virtual
  list(APPEND CUDA_NVCC_FLAGS "--generate-code arch=compute_${arch},code=compute_${arch}")
endforeach()

# Add the library target and link with OpenCV
cuda_add_library(${PROJECT_NAME} ${CUDA_SIFT_LIBRARY_TYPE} ${lib_headers} ${lib_sources})
target_link_libraries(${PROJECT_NAME})

########################################################################################
# Demo application.                                                                    #
########################################################################################
if (NOT CUDA_SIFT_HAS_PARENT)
  # OpenCV dependency: need 'core', 'imgproc', 'imgcodecs', and 'highgui'
  find_package(OpenCV COMPONENTS core imgproc imgcodecs highgui REQUIRED)
  if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is required to build the ${PROJECT_NAME} demo.")
  endif()
  if (OpenCV_VERSION VERSION_LESS 3.0.0)
    # Not sure what the OpenCV 2.x action is, warn and then likely link failure
    message(WARNING "OpenCV 3.x linkage was requested, the app may not link.  `imread` function was relocated to 'imgcodecs' module.")
  endif()

  # Include the OpenCV directories
  include_directories(${OpenCV_INCLUDE_DIRS})

  # Setup the application
  set(CUDA_SIFT_APP "cudasift-demo")
  cuda_add_executable(${CUDA_SIFT_APP} demo/mainSift.cpp demo/geomFuncs.cpp)

  # Link against libcudaSift and OpenCV
  target_link_libraries(${CUDA_SIFT_APP} ${PROJECT_NAME} ${OpenCV_LIBS})
  add_dependencies(${CUDA_SIFT_APP} ${PROJECT_NAME})

  # Copy the input test images to the binary directory (don't copy if they are building
  # the library in the root level).
  if (NOT "${CMAKE_CURRENT_BINARY_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/data")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/left.pgm" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/data")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/righ.pgm" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/data")
  endif()
endif()



# set(cuda_sources
#   # dynamic.cu
#   cudaImage.cu
#   cudaImage.h
#   cudaSiftH.cu
#   cudaSiftH.h
#   matching.cu
#   cudaSiftD.h
#   cudaSift.h
#   cudautils.h
# )

# set(sources
#   geomFuncs.cpp
#   mainSift.cpp
# )

# include_directories(
#   ${CMAKE_CURRENT_SOURCE_DIR}
# )

# SET(CUDA_SEPARABLE_COMPILATION ON)

# cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_35)

# set_target_properties(cudasift PROPERTIES
#   COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
# )

# target_link_libraries(cudasift
#   /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS}
# )

# install(FILES
#   ${cuda_sources}
#   ${sources}
#   cudaSiftD.cu
#   CMakeLists.txt
#   Copyright.txt
#   DESTINATION .
# )
# install(FILES data/left.pgm data/righ.pgm
#   DESTINATION data
# )
