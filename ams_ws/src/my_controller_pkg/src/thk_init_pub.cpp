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
#define WAIT_SHORT 1500
#define WAIT_MID 3000
#define WAIT_LONG 5000
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
	INIT_4,
	ENDLESS
} state;


int main (int argc, char** argv) {
	state = INIT_1;
	std_msgs::Float64 msg;
	my_controller_pkg::THK_AMS_msg1 xya_msg;

	ros::init(argc, argv, "thk_complete");
	ros::NodeHandle n;

	//Publisher for x-y-a coordinates
	ros::Publisher thk_xya_pub = n.advertise<my_controller_pkg::THK_AMS_msg1>("/thk_ns/thk_tiago_xya", 10);

	// Publisher for torso lift
	ros::Publisher thk_torso_pub = n.advertise<trajectory_msgs::JointTrajectory>("torso_controller/command",100);

	// Base JointTrajectory message for the torso
	trajectory_msgs::JointTrajectory traj;
	traj.joint_names.resize(1);
	traj.points.resize(1);
	traj.joint_names[0] = "torso_lift_joint";
	traj.points[0].positions.resize(1);
	traj.points[0].time_from_start = ros::Duration(2);

	// Publisher to control the gripper
	ros::Publisher thk_grip_pub = n.advertise<trajectory_msgs::JointTrajectory>("gripper_controller/command",100);

	// Base JointTrajectory message for the torso
	trajectory_msgs::JointTrajectory grip;
	grip.joint_names.resize(2);
	grip.points.resize(1);
	grip.joint_names[0] = "gripper_left_finger_joint";
	grip.joint_names[1] = "gripper_right_finger_joint";
	grip.points[0].positions.resize(2);
	grip.points[0].time_from_start = ros::Duration(2);

	// Publisher to control the arm
	ros::Publisher thk_arm_pub = n.advertise<trajectory_msgs::JointTrajectory>("arm_controller/command",100);

	// Base JointTrajectory message for the arm_controller
	trajectory_msgs::JointTrajectory arm;
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

	bool finished = false;

	/*  Simple state machine
		moves the TiaGo arm in an initial horizontal position
		lifts the torso and opens the gripper
	*/
	while (ros::ok() && finished == false) {
		switch (state) {
			case INIT_1:
				// The very first message needs to be send twice otherwise it is ignored
				thk_arm_pub.publish(arm);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_MID);
				thk_arm_pub.publish(arm);
				ros::spinOnce();
				thk_ns::wait_ms(WAIT_MID);					// wait for mechanical vibrations to disappear
				state = INIT_4;
				break;
			case INIT_4:
				traj.points[0].positions[0] = TORSO_UP;
				ROS_INFO("Initializing Torso");
				thk_torso_pub.publish(traj);

				grip.points[0].positions[0] = GRIP_OPEN;
				grip.points[0].positions[1] = GRIP_OPEN;
				ROS_INFO("Open gripper");
				thk_grip_pub.publish(grip);
				thk_ns::wait_ms(WAIT_LONG);					// wait for mechanical vibrations to disappear
				finished = true;
				break;
			case ENDLESS:
				break;
			default:
				break;
		} // switch
	} // while
} // main
