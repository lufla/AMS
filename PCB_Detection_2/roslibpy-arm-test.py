import roslibpy

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

topic = roslibpy.Topic(client, '/thk_ns/thk_tiago_xya', 'std_msgs/String')

topic.publish({"x": 0.4, "y": 0.6, "angle": 0.0})