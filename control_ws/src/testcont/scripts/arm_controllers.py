#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64

def move_joint():
	# Initialisieren des Knotens
	rospy.init_node('joint_position_controller')
	
	# Publisher für den Joint
	pub = rospy.Publisher('/arm_3_joint_position_controller/command', Float64, queue_size=10)
	
	rate = rospy.Rate(10) # 10 Hz
	while not rospy.is_shutdown():
		# Def pos
		desired_position = Float64()
		desired_position.data = 1.0
		
		pub.publish(desired_position)
		
		rospy.loginfo("Pos published")
		
		rate.sleep()

if __name__ == '__main__':
	try:
		move_joint()
	except rospy.ROSInterruptException:
		pass
