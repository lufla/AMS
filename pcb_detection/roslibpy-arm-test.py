import roslibpy
import time

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

#topic = roslibpy.Topic(client, '/thk_ns/thk_tiago_xya', 'my_controller_pkg/THK_AMS_msg1')
topic = roslibpy.Topic(client, '/thk_ns/thk_tiago_xya', 'std_msgs/String')

topic.subscribe(lambda message: client.terminate())

topic.publish({"x": 0.1, "y": 0.95, "angle": 3.14159/2})

time.sleep(1)