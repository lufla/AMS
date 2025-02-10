This file describes how to launch the demonstration of movement and positioning
on the TIAGo:

*****
Pepare every new terminal:
~$ cd ams_ws
~/ams_ws$ source devel/setup.bash
~/ams_ws$ export ROS_MASTER_URI=http://tiago-158c:11311
~/ams_ws$ export ROS_IP=<own_ip_client>
*****

Start the TIAGo and wait until TIAGo has moved into home position

Open new terminal and start thk_xya2:
~/ams_ws$ rosrun my_controller_pkg thk_arm_xya2

Open new terminal and start demonstration:
~/ams_ws$ rosrun my_controller_pkg thk_tiago2