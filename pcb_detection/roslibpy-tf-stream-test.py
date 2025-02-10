import roslibpy
import time

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

tf_stream_id = None
tf_stream_topic = None

topic_result = roslibpy.Topic(client, '/tf_stream/result', 'tf_lookup/TfStreamActionResult')

def check_tf_stream_result(message):
  global tf_stream_id
  global tf_stream_topic

  tf_stream_id = message["result"]["subscription_id"]
  tf_stream_topic = message["result"]["topic"]

  return message["result"]

topic_result.subscribe(lambda message: print(check_tf_stream_result(message)))

time.sleep(1)

topic_goal = roslibpy.Topic(client, '/tf_stream/goal', 'std_msgs/String')

topic_goal.publish(
    {
        "goal": {
          "transforms": [
            { "target": "arm_1_link", "source": "xtion_rgb_frame" }
          ],
          "subscription_id": "",
          "update": False,
        }
    }
)

while tf_stream_topic is None:
  pass

topic_stream = roslibpy.Topic(client, tf_stream_topic, 'tf/tfMessage')
topic_stream.subscribe(lambda message: print(message))

try:
    while True:
        pass
except KeyboardInterrupt:
    client.terminate()
