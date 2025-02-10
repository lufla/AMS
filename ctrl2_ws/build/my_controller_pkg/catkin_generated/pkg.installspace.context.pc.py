# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "control_toolbox;controller_interface;pluginlib;roscpp;hardware_interface;std_msgs;control_msgs;message_runtime".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lmy_controller_pkg".split(';') if "-lmy_controller_pkg" != "" else []
PROJECT_NAME = "my_controller_pkg"
PROJECT_SPACE_DIR = "/root/ctrl2_ws/install"
PROJECT_VERSION = "0.0.0"
