cmake_minimum_required(VERSION 3.10)

project(realesrgan-ov)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

find_package(OpenVINO REQUIRED)

add_executable(realesrgan-ov main.cpp)
target_link_libraries(realesrgan-ov PRIVATE openvino::runtime ${OpenCV_LIBS})
