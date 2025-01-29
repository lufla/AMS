import roslibpy
import time

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()


topic_result = roslibpy.Topic(client, '/tf_lookup/result', 'tf_lookup/TfLookupActionResult')

def check_tf_lookup_result(message, source_frame, target_frame):
  if message["result"]["transform"]["header"]["frame_id"] != target_frame:
    return None
  if message["result"]["transform"]["child_frame_id"] != source_frame:
    return None
  
  return message["result"]["transform"]["transform"]

topic_result.subscribe(lambda message: print(check_tf_lookup_result(message, "xtion_rgb_frame", "arm_1_link")))

time.sleep(1)

topic_goal = roslibpy.Topic(client, '/tf_lookup/goal', 'std_msgs/String')

topic_goal.publish(
    {
        "goal": {
          "target_frame": "arm_1_link",
          "source_frame": "xtion_rgb_frame",
          "transform_time": 1
        }
    }
)

time.sleep(1)

client.terminate()