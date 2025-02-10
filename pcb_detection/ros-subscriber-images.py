from __future__ import print_function
import base64
import time
import roslibpy
import cv2 as cv
import roslibpy.comm
import numpy as np

# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()


HEAD = 2
GRIPPER = 3

CAMERA = GRIPPER

SAVE_IMAGES = False

if CAMERA == HEAD:
    listener = roslibpy.Topic(client, '/xtion/rgb/image_raw/compressed', 'sensor_msgs/CompressedImage')
if CAMERA == GRIPPER: 
    listener = roslibpy.Topic(client, '/end_effector_camera/image_raw/compressed', 'sensor_msgs/CompressedImage')

next_frame_time = time.time_ns()

def display_image_msg(message):
    global next_frame_time
    if time.time_ns() < next_frame_time:
        return
    else:
        next_frame_time += 2 * 1E9

    base64_bytes = message['data'].encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)

    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_cv = cv.imdecode(jpg_as_np, cv.IMREAD_COLOR)
    
    
    ms = int(time.time_ns() / 1_000_000)
    if CAMERA == HEAD: camera_dir = "head"
    if CAMERA == GRIPPER: camera_dir = "gripper"
    filename = f"pcb_detection/calibration/tiago/{camera_dir}/{ms}.jpg"
    
    if SAVE_IMAGES: cv.imwrite(filename=filename, img=image_cv)

    cv.imshow("image", image_cv)

    if cv.waitKey(1) == ord('q'):
        return

def filter_message_info(message):
    result = {}
    for k, v in message.items():
        if (k == "data"):
            continue
        result[k] = v
    return result


#listener.subscribe(lambda message: print(filter_message_info(message)))
#listener.subscribe(lambda message: print(message["data"]))
listener.subscribe(display_image_msg)

try:
    while True:
        pass
except KeyboardInterrupt:
    client.terminate()