#include "ros/ros.h"
#include <my_controller_pkg/THK_AMS_msg1.h>
#include <std_msgs/Float64.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <chrono>
#include <thread>
#include <cmath>


#define GRIP_OPEN 0.03
#define GRIP_CLOSE 0.01
#define TORSO_UP 0.30
#define TORSO_DOWN 0.20
#define WAIT_SHORT 2000
#define WAIT_MID 5000
#define WAIT_LONG 10000
#define PAN1 -0.4
#define PAN2 0.4
#define TILT1 -0.3
#define TILT2 0.3

namespace thk_ns {
	/*
		sleep thread for milliseconds
	*/
	void wait_ms(int milliseconds) {
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
	}

} // namespace

enum States {
	MOVE_0,
	MOVE_1,
	MOVE_2,
	MOVE_3,
	MOVE_4,
	ENDLESS
} state;


int main (int argc, char** argv) {

	// Initialization for ROS
	ros::init(argc, argv, "thk_head");
	ros::NodeHandle n;
	
	
	// Publisher for head controller
	ros::Publisher thk_head_pub = n.advertise<trajectory_msgs::JointTrajectory>("head_controller/command",1000);
	

	// Default message for head
	trajectory_msgs::JointTrajectory head;
	head.joint_names.resize(2);
	head.points.resize(1);
	head.joint_names[0] = "head_1_joint";
	head.joint_names[1] = "head_2_joint";
	head.points[0].positions.resize(2);
	head.points[0].time_from_start = ros::Duration(1);
	
	ros::Rate loop_rate(1);
	

	
	/*  Simple state machine
		Move head around
	*/
	state = MOVE_0;
	while (ros::ok()) {
		switch (state) {
			case MOVE_0:
				/*
					Move head to home position
				*/
				head.points[0].positions[0] = 0.0;
				head.points[0].positions[1] = 0.0;
				ROS_INFO("Move head to 0.0 0.0");
				thk_head_pub.publish(head);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_MID);
				// The very first messages seems to be ignored (robot and simulation)
				thk_head_pub.publish(head);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);					// wait for mechanical vibrations to disappear
				state = MOVE_1;
				break;
			case MOVE_1:
				/*
					Move arm to first positon
				*/
				head.points[0].positions[0] = PAN1;
				head.points[0].positions[1] = TILT1;
				ROS_INFO("Moving to first position");
				thk_head_pub.publish(head);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);					// wait for mechanical vibrations to disappear
				state = MOVE_2;
				break;
			case MOVE_2:
				/*
					Move head to second positon
				*/
				head.points[0].positions[0] = PAN2;
				head.points[0].positions[1] = TILT1;
				ROS_INFO("Moving to second position");
				thk_head_pub.publish(head);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);					// wait for mechanical vibrations to disappear
				state = MOVE_3;
				break;
			case MOVE_3:
				/*
					Move arm to first positon
				*/
				head.points[0].positions[0] = PAN2;
				head.points[0].positions[1] = TILT2;
				ROS_INFO("Moving to third position");
				thk_head_pub.publish(head);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);					// wait for mechanical vibrations to disappear
				state = MOVE_4;
				break;
			case MOVE_4:
				/*
					Move arm to fourth positon
				*/
				head.points[0].positions[0] = PAN1;
				head.points[0].positions[1] = TILT2;
				ROS_INFO("Moving to fourth position");
				thk_head_pub.publish(head);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);					// wait for mechanical vibrations to disappear
				state = MOVE_1;
				break;

			case ENDLESS:
				/*
					just finish
				*/
				ROS_INFO("All done");
				break;
			default:
				break;
		} // switch
	} // while
} // main
