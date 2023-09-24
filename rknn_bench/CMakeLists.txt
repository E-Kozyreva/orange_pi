cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
if(TARGET_SOC STREQUAL "rk3588")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/3rdparty/rknn/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk3588")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  if (CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
  else()
    set(LIB_ARCH armhf)
  endif()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()


include_directories(${RKNN_API_PATH}/include)
include_directories(/usr/local/include/opencv4)
include_directories(/usr/local/include/opencv4/opencv2)

set(OpenCV_DIR "/usr/local/lib/")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video videoio highgui dnn)


#rga
if(TARGET_SOC STREQUAL "rk3588")
  set(RGA_PATH ${CMAKE_SOURCE_DIR}/3rdparty/rga/RK3588)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk3588")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RGA_LIB ${RGA_PATH}/lib/Android/${CMAKE_ANDROID_ARCH_ABI}/librga.so)
else()
  if (CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
  else()
    set(LIB_ARCH armhf)
  endif()
  set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)
endif()
include_directories( ${RGA_PATH}/include)

#Need for valgrind
SET(MYCOMPILE_FLAGS "-g")
add_definitions(${MYCOMPILE_FLAGS})

#Pthread
find_library( PTHREAD_LIBRARY NAMES "pthread")
message(STATUS "Using PThread library: " ${PTHREAD_LIBRARY})

set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov5_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_yolov5_demo
        src/main.cc
)

target_link_libraries(rknn_yolov5_demo
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
  ${PTHREAD_LIBRARY}
)


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
install(TARGETS rknn_yolov5_demo DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
