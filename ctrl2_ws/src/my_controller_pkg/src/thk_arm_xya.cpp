#include "ros/ros.h"
#include <my_controller_pkg/THK_AMS_msg1.h>
#include <std_msgs/Float64.h>
#include <chrono>
#include <thread>
#include <cmath>


namespace thk_ns {

	double x_coor = 0.0;
	double y_coor = 0.0;
	double angle = 0.0;
	bool new_data = false;
	
	// Fixed Values, need to be evaluated
	double arm1_length = 0.4;
	double arm2_length = 0.4;
	double arm3_length = 0.2;
	
	void thk_xya_callBack(const my_controller_pkg::THK_AMS_msg1::ConstPtr& msg)
	{
		x_coor = msg->x;
		y_coor = msg->y;
		angle = msg->angle;
		new_data = true;
		ROS_INFO("Data received");
	}
	
	void wait_ms(int milliseconds) {
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
	}
	
/*	bool inverseKinematics(double x, double y, double z, double& theta1, double&theta2, double& theta3)
	{
		theta1 = atan2(x,y);
		
		double wx = x - arm3_length * cos(theta1);
		double wy = y - arm3_length * sin(theta1);
		
		double a = (wx*wx + wy*wy - arm1_length*arm1_length - arm2_length*arm2_length) / (2 * arm1_length * arm2_length);
		
		if(std::abs(a) > 1.0) return false;
		
		theta3 = acos(a);
		theta2 = atan2(wy,wx) - atan2(arm2_length * sin(theta3), arm1_length + arm2_length * cos(theta3));
		
		// more needed
		return true;
	}
*/

/*	bool inverseKinematics(double x, double y, double angle1, double& theta1, double&theta2, double& theta3)
	{
		double dx = x - (arm3_length * cos(angle1));
		double dy = y - (arm3_length * sin(angle1));
		
		double c = (dx*dx + dy*dy - arm1_length*arm1_length - arm2_length*arm2_length) / ( 2 * arm1_length * arm2_length);
		
		if(std::abs(c) > 1.0) return false;
		
		theta2 = acos(c);
		
		double k1 = arm1_length + arm2_length * cos(theta2);
		double k2 = arm2_length * sin(theta2);
		
		theta1 = atan2(dy,dx) - atan2(k2,k1);
		theta3 = angle1 + theta1 - theta2 + M_PI_2; // review !!!!
		return true;
	}*/

	bool inverseKinematics(double x, double y, double theta, double* q1, double* q2, double* q3) {
    	// Die Position des Handgelenks (letzter Gelenkpunkt) bestimmt
    	double x_wrist = x - arm3_length * cos(theta);
    	double y_wrist = y - arm3_length * sin(theta);

    	// Berechnung von q2 (Mittelgelenk) unter Verwendung des Kosinussatzes
    	double D = (x_wrist * x_wrist + y_wrist * y_wrist - arm1_length*arm1_length - arm2_length*arm2_length) / (2 * arm1_length * arm2_length);
    	if (D < -1 || D > 1) {
        	// Unlösbare Position (außerhalb des erreichbaren Bereichs)
        	return false;
    	}
    
    	*q2 = atan2(sqrt(1 - D * D), D); // Winkel von q2

    	// Berechnung von q1 (Basisgelenk)
    	double k1 = arm1_length + arm2_length * cos(*q2);
    	double k2 = arm2_length * sin(*q2);
    	*q1 = atan2(y_wrist, x_wrist) - atan2(k2, k1);

    	// Berechnung von q3 (Endgelenk) unter Berücksichtigung der Orientierung theta
    	*q3 = theta - *q1 - *q2;

    	return true; // Erfolgreiche Berechnung
	}
	
} // namespace
	int main(int argc, char **argv)
	{
		ros::init(argc, argv, "THK_arm_contr");
		ros::NodeHandle n;
		
		// Subscribe to receive Messages
		ros::Subscriber sub = n.subscribe("thk_ns/thk_tiago_xya", 10, thk_ns::thk_xya_callBack);
		
		ros::Publisher thk_pub1 = n.advertise<std_msgs::Float64>("thk_ns/thk_j1_controller/command",1000);
		ros::Publisher thk_pub2 = n.advertise<std_msgs::Float64>("thk_ns/thk_j4_controller/command",1000);
		ros::Publisher thk_pub3 = n.advertise<std_msgs::Float64>("thk_ns/thk_j6_controller/command",1000);
		
		ros::Rate loop_rate(10);
		
		//double x = 1.5;
		//double y = 1.5;
		double z = 0.0;
		
		double theta1, theta2, theta3;
		std_msgs::Float64 msg;
		
		ROS_INFO("THK x-y-a running");
		
		while (ros::ok())
		{
			if (thk_ns::new_data ==  true) {
				thk_ns::new_data = false;
				if (thk_ns::inverseKinematics(thk_ns::x_coor, thk_ns::y_coor, thk_ns::angle, &theta1, &theta2, &theta3)) {
					ROS_INFO("Angles: %f : %f : %f", theta1,theta2,theta3);
					msg.data = theta1 + M_PI_2;
					thk_pub1.publish(msg);
					ros::spinOnce();
					msg.data = theta2;
					thk_pub2.publish(msg);
					ros::spinOnce();
					msg.data = theta3;
					thk_pub3.publish(msg);
					ros::spinOnce();
				} else
					ROS_INFO("No calc possible");
			
			}
		
		
			ros::spinOnce();
			
			loop_rate.sleep();
		}
		
		return 0;	
	} // main


