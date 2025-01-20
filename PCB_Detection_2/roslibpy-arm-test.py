import roslibpy

# https://docs.pal-robotics.com/tiago-single/handbook.html#jointcommander-plugin-configuration
# https://docs.pal-robotics.com/tiago-single/handbook.html#arm-control-ros-api
# https://docs.pal-robotics.com/tiago-single/handbook.html#controller-manager-services

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

listener = roslibpy.Topic(client, '/arm_controller/state', 'std_msgs/String')
listener.subscribe(lambda message: print(message))

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