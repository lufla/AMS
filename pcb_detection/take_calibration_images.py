import time
import cv2 as cv
from dotenv import dotenv_values

config = dotenv_values(".env")

cap = cv.VideoCapture(int(config["camera_index"]))  # Change the index if you have multiple cameras

#images = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting.")
        break

    #images.append(frame)
    #print(len(images))

    cv.imshow('frame', frame)

    ms = int(time.time_ns() / 1_000_000)
    #print(ms)
    ret_write = cv.imwrite(filename=f"pcb_detection/calibration/webcam/{ms}.png", img=frame)
    if not ret_write:
        print("Can not write image")
        exit()

    time.sleep(2)

    # Break loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()