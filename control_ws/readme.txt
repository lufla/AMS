For Testing:
- start gazebo to start controller manager:
    roslaunch tiago_158_gazebo tiago_gazebo.launch
- stop arm_controller to prevent resource conflicts:
    rosservice call /controller_manager/switch_controller "{start_controllers: [], stop_controllers: [arm_controller], strictness: 2}" 

- start own controller:
    roslaunch testcont joint_position_controller.launch
- publish positions of arm_3_joint to controller:
    rostopic pub /joint_position_controller/command std_msgs/Float64 "data: -1.0"

