from __future__ import print_function
import roslibpy

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

listener = roslibpy.Topic(client, '/tf_lookup/result', 'tf_lookup/TfLookupActionResult')
#listener = roslibpy.Topic(client, '/xtion/rgb/image_raw', 'std_msgs/String')

listener.subscribe(lambda message: print('Heard talking: ' + message))

try:
    while True:
        pass
except KeyboardInterrupt:
    client.terminate()