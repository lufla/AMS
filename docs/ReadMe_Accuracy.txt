This file describes all steps to be done in order to demonstrate
the accuracy of the own controller:

****
Prepare every new terminal:
~$ cd ams_ws/
~/ams_ws$ source devel/setup.bash
****

Open new terminal and start Gazebo simulation:
~/ams_ws$ roslaunch tiago_158_gazebo tiago_gazebo.launch
!! wait for tuck_arm to finish by moving arm in home position !!
!! log: '[tuck_arm-...finished cleanly' !!

Open new terminal to stop and start controllers:
~/ams_ws$ rosservice call controller_manager/switch_controller "{start_controllers: [], stop_controllers:[arm_controller], strictness: 2}"
~/ams_ws$ roslaunch my_controller_pkg my_controller.launch

Open new terminal to run publisher/subscriber for coordinate transform
~/ams_ws$ rosrun my_controller_pkg thk_arm_xya

Open new terminal to run accuracy demonstration
~/ams_ws$ rosrun my_controller_pkg thk_accuracy
