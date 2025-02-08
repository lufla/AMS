my_controller_pkg/  
&emsp;	config/  
&emsp;&emsp;		controller.yaml	&emsp;&emsp;&emsp;		Configuration for own controllers  
&emsp;	launch/  
&emsp;&emsp;		my_controller.launch&emsp;&emsp;&emsp;		Launch file to start own controllers  
&emsp;	msg/  
&emsp;&emsp;		THK_AMS_msg1.msg&emsp;&emsp;&emsp;		Definition of xya-message type  
&emsp;	src/  
&emsp;&emsp;		my_controller.cpp&emsp;		Code of the own controller  
&emsp;&emsp;		thk_accuracy.cpp&emsp;		Code for gazebo simulation of own controller  
&emsp;&emsp;		thk_arm_xya.cpp	&emsp;		Publisher/Subscriber for inverse kinematics own controller  
&emsp;&emsp;		thk_arm_xya2.cpp&emsp;		Publisher/Subscriber for inverse kinematics PAL arm_controller  
&emsp;&emsp;		thk_init_pub.cpp&emsp;		Publisher to move arm in horizontal position, PAL arm_controller  
&emsp;&emsp;		thk_head.cpp&emsp;			Publisher to demonstrate head movement PAL head_controller  
&emsp;&emsp;		thk_tiago.cpp&emsp;			Gazebo demonstration of arm movement own controller  
&emsp;&emsp;		thk_tiago2.cpp&emsp;			Demonstration of arm movement PAL arm_controller  
&emsp;	CMakeLists.txt		&emsp;&emsp;		CMake file  
&emsp;	package.xml		&emsp;&emsp;		Definitions for packet dependencies  
&emsp;	thk_controllers_plugins.xml	&emsp;&emsp;	Generated controller configuration  
