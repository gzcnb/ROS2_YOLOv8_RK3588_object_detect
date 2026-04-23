cmake_minimum_required(VERSION 3.8)
project(yolov8_detect_rknn_pkg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ========== 1. 查找 ROS2 依赖 ==========
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# ========== 2. 查找 OpenCV ==========
find_package(OpenCV REQUIRED)
# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS})

#图片检测
add_executable(img_detect_node src/img_detect_node.cpp)
target_include_directories(img_detect_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(
  img_detect_node
  "rclcpp"
  "std_msgs"
  "geometry_msgs"
)
target_link_libraries(img_detect_node ${OpenCV_LIBS} /lib/librknnrt.so)

#视频检测
add_executable(video_detect_node src/video_detect_node.cpp)
target_include_directories(video_detect_node 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(
  video_detect_node
  "rclcpp"
  "std_msgs"
  "geometry_msgs"
)
target_link_libraries(video_detect_node ${OpenCV_LIBS} /lib/librknnrt.so)
 


install(TARGETS 
  img_detect_node
  video_detect_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
