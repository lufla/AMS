
#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.hpp>
#include <realtime_tools/realtime_publisher.h>
#include <std_msgs/Float64.h>
#include <control_toolbox/pid.h>

namespace thk_ns {
class MyController : public controller_interface::Controller<hardware_interface::EffortJointInterface>
{
	public:
	  struct Commands
	  {
	  	double position_;
	  	double velocity_;
	  	bool has_velocity_;
	  };
	  
	  void setCommand(double pos_command)
	  {
	  	command_struct_.position_ = pos_command;
	  	command_struct_.has_velocity_ = false;
	  	
	  	command_.writeFromNonRT(command_struct_);
	  }
	  
	  void setCommand(double pos_command, double vel_command)
	  {
	  	command_struct_.position_ = pos_command;
	  	command_struct_.velocity_ = vel_command;
	  	command_struct_.has_velocity_ = true;
	  	
	  	command_.writeFromNonRT(command_struct_);
	  }
	  
	  bool init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle& nh) override
	  {
	    //joint_ = hw->getHandle("arm_3_joint");
	    std::string joint_name;
	    if (!nh.getParam("joint", joint_name))
	    {
	    	ROS_ERROR("No joint given (namespace: %s)", nh.getNamespace().c_str());
	    	return false;
	    }
	    
	    if (!pid_controller_.init(ros::NodeHandle(nh, "pid")))
	    	return false;
	   
	    joint_ = hw->getHandle(joint_name);
	    
	    command_sub_ = nh.subscribe<std_msgs::Float64>("command", 1, &MyController::commandCB, this);
	    return true;
	  }
  
	  void update(const ros::Time& time, const ros::Duration& period) override
	  {
	    command_struct_ = *(command_.readFromRT());
	    double command_position = command_struct_.position_;
	    double command_velocity = command_struct_.velocity_;
	    bool has_velocity_ = command_struct_.has_velocity_;
	    
	    double error, vel_error;
	    double commanded_effort;
	    
	    double current_position = joint_.getPosition();
	    
	    error = command_position - current_position;
	    
	    commanded_effort = pid_controller_.computeCommand(error,period);
	    
	    joint_.setCommand(commanded_effort);
	  }
  
	  void starting(const ros::Time& /*time*/) override
	  {
	    double pos_command = joint_.getPosition();
	    
	    command_struct_.position_ = pos_command;
	    command_struct_.has_velocity_ = false;
	    
	    command_.initRT(command_struct_);
	    
	    pid_controller_.reset();
	  }
  
	  void stopping(const ros::Time& /*time*/) override
	  {}

	  void commandCB(const std_msgs::Float64ConstPtr& msg)
	  {
	    //command_ = msg->data;
	    setCommand(msg->data);
	  }
	
	private:
	  hardware_interface::JointHandle joint_;
	  //double command_;
	  realtime_tools::RealtimeBuffer<Commands> command_;
	  Commands command_struct_;
	  control_toolbox::Pid pid_controller_;
	  ros::Subscriber command_sub_;
	};
	
} // namespace

PLUGINLIB_EXPORT_CLASS(thk_ns::MyController, controller_interface::ControllerBase)

