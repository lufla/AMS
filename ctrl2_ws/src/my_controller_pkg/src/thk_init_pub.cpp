#include "ros/ros.h"
#include "std_msgs/String.h"
#include <std_msgs/Float64.h>
#include <chrono>
#include <thread>

#include <sstream>

void wait_ms(int milliseconds) {
	std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

int main(int argc, char **argv)
{
	ros::init(argc,argv, "thk_init_pub");
	ros::NodeHandle n;
	
	ros::Publisher thk_init_pub1 = n.advertise<std_msgs::Float64>("/thk_ns/thk_j3_controller/command",10);
	ros::Publisher thk_init_pub2 = n.advertise<std_msgs::Float64>("/thk_ns/thk_j2_controller/command",10);
	ros::Publisher thk_init_pub3 = n.advertise<std_msgs::Float64>("/thk_ns/thk_j7_controller/command",10);
	
	ros::Rate loop_rate(1);
	
	int count = 0;
	float data_value = -1.57;
	
/*	while (ros::ok())
	{
		std_msgs::String msg;
		std::stringstream ss;
		ss << "hello world " << count;
		
		msg.data = ss.str();
		ROS_INFO("%s", msg.data.c_str());
		
		chatter_pub.publish(msg);
		
		ros::spinOnce();
		
		loop_rate.sleep();
		++count;
	} */
	
	std_msgs::Float64 msg;
	msg.data = -1.57;
	ROS_INFO("Initializing Joint 3");
	thk_init_pub1.publish(msg);
	ros::spinOnce();
	wait_ms(5000);
	
	msg.data = 0.00;
	ROS_INFO("Initializing Joint 2");
	thk_init_pub2.publish(msg);
	ros::spinOnce();
	wait_ms(5000);
	
	msg.data = -1.57;
	ROS_INFO("Initializing Joint 7");
	thk_init_pub3.publish(msg);
	ros::spinOnce();
	wait_ms(5000);
	
	return 0;
}
