import time
import roslibpy

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

talker = roslibpy.Topic(client, '/chatter', 'std_msgs/String')

try:
    while client.is_connected:
        talker.publish(roslibpy.Message({'data': 'Hello World!'}))
        print('Sending message...')
        time.sleep(1)
except KeyboardInterrupt:
    client.terminate()

talker.unadvertise()

client.terminate()