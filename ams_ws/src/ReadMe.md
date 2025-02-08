my_controller_pkg/  
&emsp;	config/  
&emsp;&emsp;		controller.yaml			Cofiguration for own controllers  
&emsp;	launch/  
&emsp;&emsp;		my_controller.launch		Launch file to start own controllers  
&emsp;	msg/  
&emsp;&emsp;		THK_AMS_msg1.msg		Definition of xya-message type  
	src/  
		my_controller.cpp		Code of the own controller  
		thk_accuracy.cpp		Code for gazebo simulation of own controller  
		thk_arm_xya.cpp			Publisher/Subscriber for inverse kinematics own controller  
		thk_arm_xya2.cpp		Publisher/Subscriber for inverse kinematics PAL arm_controller  
		thk_init_pub.cpp		Publisher to move arm in horizontal position, PAL arm_controller  
		thk_head.cpp			Publisher to demonstrate head movement PAL head_controller  
		thk_tiago.cpp			Gazebo demonstration of arm movement own controller  
		thk_tiago2.cpp			Demonstration of arm movement PAL arm_controller  
	CMakeLists.txt				CMake file  
	package.xml				Definitions for packet dependencies  
	thk_controllers_plugins.xml		Generated controller configuration  
