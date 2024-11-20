#include "my_controller_controller/my_controller.h"

namespace my_controller_controller {

bool MyController::init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle& nh) {
  std::string joint_name;
  if (!nh.getParam("joint_name", joint_name)) {
    ROS_ERROR("Parameter 'joint_name' not set");
    return false;
  }

  joint_ = hw->getHandle(joint_name);
  command_ = 1.0; // Example: set a constant command effort
  return true;
}

void MyController::starting(const ros::Time& time) {
  ROS_INFO("Starting the controller");
}

void MyController::update(const ros::Time& time, const ros::Duration& period) {
  joint_.setCommand(command_);
}

void MyController::stopping(const ros::Time& time) {
  joint_.setCommand(0.0); // Stop movement
  ROS_INFO("Stopping the controller");
}

} // namespace my_controller_controller

PLUGINLIB_EXPORT_CLASS(my_controller_controller::MyController, controller_interface::ControllerBase)
