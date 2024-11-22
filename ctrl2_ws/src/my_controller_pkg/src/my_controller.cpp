
#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.hpp>
#include <realtime_tools/realtime_publisher.h>
#include <std_msgs/Float64.h>

namespace thk_ns {
class MyController : public controller_interface::Controller<hardware_interface::EffortJointInterface>
{
	public:
	  bool init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle& nh) override
	  {
	    joint_ = hw->getHandle("arm_3_joint");
	    command_sub_ = nh.subscribe<std_msgs::Float64>("command", 1, &MyController::commandCB, this);
	    return true;
	  }
  
	  void update(const ros::Time& /*time*/, const ros::Duration& /*period*/) override
	  {
	    joint_.setCommand(command_);
	  }
  
	  void starting(const ros::Time& /*time*/) override
	  {
	    command_ = 0.0;
	  }
  
	  void stopping(const ros::Time& /*time*/) override
	  {}

	  void commandCB(const std_msgs::Float64ConstPtr& msg)
	  {
	    command_ = msg->data;
	  }
	
	private:
	  hardware_interface::JointHandle joint_;
	  double command_;
	  ros::Subscriber command_sub_;
	};
	
} // namespace

PLUGINLIB_EXPORT_CLASS(thk_ns::MyController, controller_interface::ControllerBase)

