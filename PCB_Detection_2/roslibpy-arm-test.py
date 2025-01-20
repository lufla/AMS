import roslibpy

# https://docs.pal-robotics.com/tiago-single/handbook.html#jointcommander-plugin-configuration
# https://docs.pal-robotics.com/tiago-single/handbook.html#arm-control-ros-api
# https://docs.pal-robotics.com/tiago-single/handbook.html#controller-manager-services

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

topic = roslibpy.Topic(client, '/thk_ns/thk_tiago_xya', 'std_msgs/String')

topic.publish({"x": 0.4, "y": 0.4, "angle": 0.0})
