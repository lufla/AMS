#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from control_msgs.msg import JointControllerState

class JointPositionController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('joint_position_controller')
        
        # Read parameters
        self.joint_name = rospy.get_param('~joint', 'default_joint')
        self.p_gain = rospy.get_param('~p_gain', 1.0)
        self.i_gain = rospy.get_param('~i_gain', 0.0)
        self.d_gain = rospy.get_param('~d_gain', 0.0)
        
        # Setpoint and state variables
        self.set_point = 0.0
        self.current_position = 0.0
        self.error = 0.0
        self.commanded_effort = 0.0
        
        # ROS publishers and subscribers
        self.command_sub = rospy.Subscriber('command', Float64, self.set_command_cb)
        self.state_pub = rospy.Publisher('state', JointControllerState, queue_size=1)
        
        # PID error terms
        self.prev_error = 0.0
        self.integral = 0.0
    
    def set_command_cb(self, msg):
        self.set_point = msg.data
    
    def update(self):
        # Compute error
        self.error = self.set_point - self.current_position

        # PID computations
        derivative = (self.error - self.prev_error)
        self.integral += self.error
        self.commanded_effort = (self.p_gain * self.error +
                                 self.i_gain * self.integral +
                                 self.d_gain * derivative)
        
        # Update previous error
        self.prev_error = self.error
        
        # Here you would send the commanded_effort to the joint, for simplicity we'll just print it
        rospy.loginfo(f"Commanded Effort: {self.commanded_effort}")

        # Publish state
        state_msg = JointControllerState()
        state_msg.set_point = self.set_point
        state_msg.process_value = self.current_position
        state_msg.error = self.error
        state_msg.command = self.commanded_effort
        self.state_pub.publish(state_msg)
        
    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = JointPositionController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
