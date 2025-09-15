cmake_minimum_required (VERSION 3.8)

project ("digits-recognition-model")

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR thirdparty/OpenCV_4_12_0/opencv/build)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable (digits-recognition-model "digits-recognition-model.cpp")

set(OPENCV_DIR "thirdparty/OpenCV_4_12_0/opencv/build/x64/vc16")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/test_image.jpg
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(digits-recognition-model ${OpenCV_LIBS} )
