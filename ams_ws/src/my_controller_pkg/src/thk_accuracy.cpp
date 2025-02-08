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
#define ROW1 0.3
#define ROW2 0.5
#define COL1 0.3
#define COL2 0.6
#define STEP 0.0001			// smallest step to move arm in meter

namespace thk_ns {
	/*
		sleep thread for milliseconds
	*/
	void wait_ms(int milliseconds) {
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
	}

} // namespace

enum States {
	INIT_1,
	INIT_2,
	INIT_3,
	INIT_4,
	MOVE_1,
	MOVE_2,
	ENDLESS
} state;


int main (int argc, char** argv) {

	// Initialization for ROS
	ros::init(argc, argv, "thk_complete");
	ros::NodeHandle n;
	
	// Publish to controllers that are just used in initializing
	ros::Publisher thk_init_pub1 = n.advertise<std_msgs::Float64>("/thk_ns/thk_j3_controller/command",100);
	ros::Publisher thk_init_pub2 = n.advertise<std_msgs::Float64>("/thk_ns/thk_j2_controller/command",100);
	ros::Publisher thk_init_pub3 = n.advertise<std_msgs::Float64>("/thk_ns/thk_j7_controller/command",100);
	std_msgs::Float64 msg;
	
	//Publisher for x-y-a coordinates
	ros::Publisher thk_xya_pub = n.advertise<my_controller_pkg::THK_AMS_msg1>("/thk_ns/thk_tiago_xya", 10);
	my_controller_pkg::THK_AMS_msg1 xya_msg;
	
	// Publisher for torso lift
	ros::Publisher thk_torso_pub = n.advertise<trajectory_msgs::JointTrajectory>("torso_controller/command",1000);
	
	// Default message for torso
	trajectory_msgs::JointTrajectory traj;
	traj.joint_names.resize(1);
	traj.points.resize(1);
	traj.joint_names[0] = "torso_lift_joint";
	traj.points[0].positions.resize(1);
	traj.points[0].time_from_start = ros::Duration(1);
	
	// Publisher to control the gripper
	ros::Publisher thk_grip_pub = n.advertise<trajectory_msgs::JointTrajectory>("gripper_controller/command",100);
	
	// Default message for gripper
	trajectory_msgs::JointTrajectory grip;
	grip.joint_names.resize(2);
	grip.points.resize(1);
	grip.joint_names[0] = "gripper_left_finger_joint";
	grip.joint_names[1] = "gripper_right_finger_joint";
	grip.points[0].positions.resize(2);
	grip.points[0].time_from_start = ros::Duration(1);
	
	ros::Rate loop_rate(1);
	
	double move = STEP;
	bool in_progress = true;
	
	
	/*  Simple state machine
		Move arm to horizontal position
		Than move arm by small steps
	*/
	state = INIT_1;
	while (ros::ok() && (in_progress == true)) {
		switch (state) {
			case INIT_1:
				/*
					From home position, move 'ellbow' joint
				*/
				msg.data = -1 * M_PI_2;
				ROS_INFO("Initializing Joint 3");
				thk_init_pub1.publish(msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_MID);
				// The very first messages seems to be ignored (robot and simulation)
				thk_init_pub1.publish(msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = INIT_2;
				break;
			case INIT_2:
				/*
					Move arm to horizontal position
				*/
				msg.data = 0.00;
				ROS_INFO("Initializing Joint 2");
				thk_init_pub2.publish(msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = INIT_3;
				break;
			case INIT_3:
				/*
					Turn gripper to straight down
				*/
				msg.data = -1 * M_PI_2;
				ROS_INFO("Initializing Joint 7");
				thk_init_pub3.publish(msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = INIT_4;
				break;
			case INIT_4:
				/*
					Lift torso and open gripper
				*/
				traj.points[0].positions[0] = TORSO_UP;
				ROS_INFO("Initializing Torso");
				thk_torso_pub.publish(traj);

				grip.points[0].positions[0] = GRIP_OPEN;
				grip.points[0].positions[1] = GRIP_OPEN;
				ROS_INFO("Open gripper");
				thk_grip_pub.publish(grip);
				thk_ns::wait_ms(WAIT_LONG);
				state = MOVE_1;
				break;
			case MOVE_1:
				/*
					Move arm to initialize positon
				*/
				xya_msg.x = ROW1;
				xya_msg.y = COL1;
				xya_msg.angle = M_PI_2;
				ROS_INFO("Moving to first position");
				thk_xya_pub.publish (xya_msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = MOVE_2;
				break;
			case MOVE_2:
				/*
					Move arm in one direction by STEP every WAIT_SHORT milliseconds
				*/
				xya_msg.x = ROW1;
				xya_msg.y = COL1 + move;
				xya_msg.angle = M_PI_2;
				ROS_INFO("Moving Y by %f to %f",STEP, (COL1+move));
				thk_xya_pub.publish (xya_msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_SHORT);
				move += STEP;
				if (move > 0.1) state = ENDLESS;
				break;

			case ENDLESS:
				/*
					just finish
				*/
				ROS_INFO("All done");
				in_progress = false;
				break;
			default:
				break;
		} // switch
	} // while
} // main
