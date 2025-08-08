cmake_minimum_required(VERSION 3.10)
project(LinearRegressionCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(include)

add_executable(linear_regression
    main.cpp
    src/dataset.cpp
    src/linear_regression.cpp
    src/utils.cpp
)

target_link_libraries(linear_regression Eigen3::Eigen)
