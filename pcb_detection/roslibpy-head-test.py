import roslibpy
import time

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

# rosnode kill /pal_head_manager

topic = roslibpy.Topic(client, '/head_controller/command', 'std_msgs/String')

topic.subscribe(lambda message: client.terminate())

# head_1_joint pan left right, head_2_joint tilt up down

topic.publish(
    {
        "joint_names":  ["head_1_joint", "head_2_joint"],
        "points": [{"positions": [0.0, -0.85], "time_from_start": {"secs": 1}}]
    }
)

time.sleep(1)

"""
header: 
  seq: 2
  stamp: 
    secs: 0
    nsecs:         0
  frame_id: ''
joint_names: 
  - head_1_joint
  - head_2_joint
points: 
  - 
    positions: [0.0026, 0.3193]
    velocities: []
    accelerations: []
    effort: []
    time_from_start: 
      secs: 1
      nsecs:         0
"""