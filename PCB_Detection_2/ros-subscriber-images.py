from __future__ import print_function
import base64
import time
import roslibpy
import cv2 as cv
import roslibpy.comm
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image

# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

from dotenv import dotenv_values
config = dotenv_values(".env")

client = roslibpy.Ros(host=config["ros_host"], port=9090)
client.run()

bridge = CvBridge()

#listener = roslibpy.Topic(client, '/xtion/rgb/image_raw/compressed', 'sensor_msgs/CompressedImage')
#listener = roslibpy.Topic(client, '/end_effector_camera/image_raw/compressed', 'sensor_msgs/CompressedImage')
listener = roslibpy.Topic(client, '/xtion/depth_registered/image_raw/compressed', 'sensor_msgs/CompressedImage')

#listener.subscribe(lambda message: print(filter_message_info(message)))
#listener.subscribe(lambda message: print(message["data"]))
listener.subscribe(lambda message: display_image_msg(message))

def display_image_msg(message):
    image_value = CompressedImage()
    image_value.header.seq = message['header']['seq']
    image_value.header.stamp.secs = message['header']['stamp']['secs']
    image_value.header.stamp.nsecs = message['header']['stamp']['nsecs']
    image_value.header.frame_id = message['header']['frame_id']
    #image_value.height = message['height']
    #image_value.width = message['width']
    #image_value.encoding = message['encoding']
    #image_value.is_bigendian = message['is_bigendian']
    #image_value.step = message['step']
    image_value.format = message['format']

    base64_bytes = message['data'].encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)
    image_value.data = image_bytes
    print("*")


    with open("image.jpg" , 'wb') as image_file:
        image_file.write(image_bytes)
    image_cv = cv.imread("image.jpg")

    #image_cv = bridge.imgmsg_to_cv2(image_value, desired_encoding='passthrough')

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

try:
    while True:
        pass
except KeyboardInterrupt:
    client.terminate()