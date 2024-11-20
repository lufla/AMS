#ifndef MY_CONTROLLER_H
#define MY_CONTROLLER_H

#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace my_controller_controller {

class MyController : public controller_interface::Controller<hardware_interface::EffortJointInterface> {
private:
  hardware_interface::JointHandle joint_;
  double command_;

public:
  MyController() : command_(0.0) {}

  bool init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle& nh) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void stopping(const ros::Time& time) override;
};

} // namespace my_controller_controller

#endif // MY_CONTROLLER_H
