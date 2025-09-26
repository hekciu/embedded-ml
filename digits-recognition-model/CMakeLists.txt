cmake_minimum_required (VERSION 3.8)

project ("digits-recognition-model")

set(CMAKE_CXX_STANDARD 17)

add_executable (digits-recognition-model "digits-recognition-model.cpp" "dataset.cpp" "model-training.cpp")

# opencv
set(OpenCV_DIR thirdparty/OpenCV_4_12_0/opencv/build)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS} )

target_link_libraries(digits-recognition-model PRIVATE ${OpenCV_LIBS} )

# tensorflow
#add_library(tensorflow STATIC ${CMAKE_SOURCE_DIR}/thirdparty/TensorFlow_2_18_1/lib/tensorflow.lib)
add_library(tensorflow STATIC IMPORTED)

#add_dependencies(digits-recognition-model tensorflow)

#set_target_properties(tensorflow PROPERTIES LINKER_LANGUAGE C IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/thirdparty/TensorFlow_2_18_1/lib/tensorflow.lib)
set_target_properties(tensorflow PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/thirdparty/TensorFlow_2_18_1/lib/tensorflow.lib
)

include_directories(${CMAKE_SOURCE_DIR}/thirdparty/Tensorflow_2_18_1/include)
#link_directories(${CMAKE_SOURCE_DIR}/thirdparty/Tensorflow_2_18_1/lib)

target_link_libraries(digits-recognition-model PRIVATE tensorflow)


# copy assets
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/test_image.jpg
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/Train.csv
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/graph.pb
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
