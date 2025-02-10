#include "ros/ros.h"
#include <my_controller_pkg/THK_AMS_msg1.h>
#include <std_msgs/Float64.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <chrono>
#include <thread>
#include <cmath>

#define GRIP_OPEN 0.03
#define GRIP_CLOSE 0.02
#define TORSO_UP 0.30
#define TORSO_DOWN 0.20
#define WAIT_SHORT 2000
#define WAIT_MID 5000
#define WAIT_LONG 10000
#define ROW1 0.32
#define ROW2 0.5
#define COL1 0.32
#define COL2 0.6
#define JOINT3INIT -1.57
#define JOINT2INIT 0.00
#define JOINT7INIT -1.57

namespace thk_ns {
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
	MOVE_1a,
	MOVE_2,
	MOVE_3,
	MOVE_3a,
	MOVE_4,
	ENDLESS
} state;


int main (int argc, char** argv) {
	state = INIT_1;
	std_msgs::Float64 msg;
	my_controller_pkg::THK_AMS_msg1 xya_msg;

	ros::init(argc, argv, "thk_complete");
	ros::NodeHandle n;
	
	// Publish to controllers that are just used in initializing
	//ros::Publisher thk_init_pub1 = n.advertise<std_msgs::Float64>("/thk_ns/thk_j3_controller/command",100);
	//ros::Publisher thk_init_pub2 = n.advertise<std_msgs::Float64>("/thk_ns/thk_j2_controller/command",100);
	//ros::Publisher thk_init_pub3 = n.advertise<std_msgs::Float64>("/thk_ns/thk_j7_controller/command",100);
	
	//Publisher for x-y-a coordinates
	ros::Publisher thk_xya_pub = n.advertise<my_controller_pkg::THK_AMS_msg1>("/thk_ns/thk_tiago_xya", 10);
	
	// Publisher for torso lift
	ros::Publisher thk_torso_pub = n.advertise<trajectory_msgs::JointTrajectory>("torso_controller/command",1000);
	trajectory_msgs::JointTrajectory traj;
	//traj.header.stamp = ros::Time::now();
	//traj.header.frame_id = "base_link";
	traj.joint_names.resize(1);
	traj.points.resize(1);
	traj.joint_names[0] = "torso_lift_joint";
	traj.points[0].positions.resize(1);
	traj.points[0].time_from_start = ros::Duration(2);
	
	// Publisher to control the gripper
	ros::Publisher thk_grip_pub = n.advertise<trajectory_msgs::JointTrajectory>("gripper_controller/command",100);
	trajectory_msgs::JointTrajectory grip;
	//grip.header.stamp = ros::Time::now();
	//grip.header.frame_id = "base_link";
	grip.joint_names.resize(2);
	grip.points.resize(1);
	grip.joint_names[0] = "gripper_left_finger_joint";
	grip.joint_names[1] = "gripper_right_finger_joint";
	grip.points[0].positions.resize(2);
	grip.points[0].time_from_start = ros::Duration(2);

	ros::Publisher thk_arm_pub = n.advertise<trajectory_msgs::JointTrajectory>("arm_controller/command",100);
	trajectory_msgs::JointTrajectory arm;
	//arm.header.stamp = ros::Time::now();
	//arm.header.frame_id = "base_link";
	arm.joint_names.resize(7);
	arm.points.resize(1);
	arm.joint_names[0] = "arm_1_joint";
	arm.joint_names[1] = "arm_2_joint";
	arm.joint_names[2] = "arm_3_joint";
	arm.joint_names[3] = "arm_4_joint";
	arm.joint_names[4] = "arm_5_joint";
	arm.joint_names[5] = "arm_6_joint";
	arm.joint_names[6] = "arm_7_joint";
	arm.points[0].positions.resize(7);
	arm.points[0].time_from_start = ros::Duration(3);
	arm.points[0].positions[0] = 0.0;
	arm.points[0].positions[1] = JOINT2INIT;
	arm.points[0].positions[2] = JOINT3INIT;
	arm.points[0].positions[3] = M_PI_2;
	arm.points[0].positions[4] = -M_PI_2;
	arm.points[0].positions[5] = 0.0;
	arm.points[0].positions[6] = JOINT7INIT;
	
	ros::Rate loop_rate(1);
	
	
	/*  Simple state machine
		just circulates between 4 hardcoded points
		pick-up and release just implied
	*/
	
	while (ros::ok()) {
		switch (state) {
			case INIT_1:
				/*msg.data = -1.57;
				ROS_INFO("Initializing Joint 3");
				thk_init_pub1.publish(msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_MID);
				thk_init_pub1.publish(msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);*/
				
				//arm.header.stamp = ros::Time::now();
				thk_arm_pub.publish(arm);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_MID);
				//arm.header.stamp = ros::Time::now();
				thk_arm_pub.publish(arm);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_MID);
				state = INIT_4;
				break;
			case INIT_2:
				msg.data = 0.00;
				ROS_INFO("Initializing Joint 2");
				//thk_init_pub2.publish(msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = INIT_3;
				break;
			case INIT_3:
				msg.data = -1.57;
				ROS_INFO("Initializing Joint 7");
				//thk_init_pub3.publish(msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = INIT_4;
				break;
			case INIT_4:
	//traj.header.stamp = ros::Time::now();
				traj.points[0].positions[0] = TORSO_UP;
				ROS_INFO("Initializing Torso");
				thk_torso_pub.publish(traj);

	//grip.header.stamp = ros::Time::now();
				grip.points[0].positions[0] = GRIP_OPEN;
				grip.points[0].positions[1] = GRIP_OPEN;
				ROS_INFO("Open gripper");
				thk_grip_pub.publish(grip);
				thk_ns::wait_ms(WAIT_LONG);
				state = MOVE_1;
				break;
			case MOVE_1:
				xya_msg.x = ROW1;
				xya_msg.y = COL1;
				xya_msg.angle = M_PI_2;
				ROS_INFO("Moving to first position");
				thk_xya_pub.publish (xya_msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = MOVE_1a;
				break;
			case MOVE_1a:
				traj.points[0].positions[0] = TORSO_DOWN;
				ROS_INFO("Lowering torso");
				thk_torso_pub.publish(traj);
				thk_ns::wait_ms(WAIT_MID);

	//grip.header.stamp = ros::Time::now();
				grip.points[0].positions[0] = GRIP_CLOSE;
				grip.points[0].positions[1] = GRIP_CLOSE;
				ROS_INFO("Closing gripper");
				thk_grip_pub.publish(grip);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_MID);

				traj.points[0].positions[0] = TORSO_UP;
				ROS_INFO("Lifting torso");
	//traj.header.stamp = ros::Time::now();
				thk_torso_pub.publish(traj);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_MID);
				state = MOVE_2;
				break;
			case MOVE_2:
				xya_msg.x = ROW2;
				xya_msg.y = COL1;
				xya_msg.angle = M_PI_2;
				ROS_INFO("Moving to second position");
				thk_xya_pub.publish (xya_msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = MOVE_3;
				break;
			case MOVE_3:
				xya_msg.x = ROW2;
				xya_msg.y = COL2;
				xya_msg.angle = M_PI_2;
				ROS_INFO("Moving to third position");
				thk_xya_pub.publish (xya_msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = MOVE_3a;
				break;
			case MOVE_3a:
				traj.points[0].positions[0] = TORSO_DOWN;
				ROS_INFO("Lowering torso");
				thk_torso_pub.publish(traj);

	//grip.header.stamp = ros::Time::now();
				grip.points[0].positions[0] = GRIP_OPEN;
				grip.points[0].positions[1] = GRIP_OPEN;
				ROS_INFO("Opening gripper");
				thk_grip_pub.publish(grip);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);

	//traj.header.stamp = ros::Time::now();
				traj.points[0].positions[0] = TORSO_UP;
				ROS_INFO("Lifting torso");
				thk_torso_pub.publish(traj);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = MOVE_4;
				break;
			case MOVE_4:
				xya_msg.x = ROW1;
				xya_msg.y = COL2;
				xya_msg.angle = M_PI_2;
				ROS_INFO("Moving to fourth position");
				thk_xya_pub.publish (xya_msg);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_LONG);
				state = MOVE_1;
				break;
			case ENDLESS:
				break;
			default:
				break;
		} // switch
	} // while
} // main
