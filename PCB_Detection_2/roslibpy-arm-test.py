import roslibpy
from _THK_AMS_msg1 import THK_AMS_msg1
import rospy
from io import StringIO


# https://docs.pal-robotics.com/tiago-single/handbook.html#jointcommander-plugin-configuration
# https://docs.pal-robotics.com/tiago-single/handbook.html#arm-control-ros-api
# https://docs.pal-robotics.com/tiago-single/handbook.html#controller-manager-services

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

topic = roslibpy.Topic(client, '/thk_ns/thk_tiago_xya', 'std_msgs/String')

message = THK_AMS_msg1()
message.x = 0.4
message.y = 0.4
message.angle = 0.0

rospy.msg.serialize_message()

#topic.publish({"x": 0.4, "y": 0.4, "angle": 0.0})
topic.publish(message.serialize())

"""
topics = client.get_topics()
topics.sort()
for topic in topics:
    if "arm" in topic:
        print(topic)

try:
    while True:
        pass
except KeyboardInterrupt:
    client.terminate()
"""