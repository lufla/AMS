from time import sleep

import numpy as np
import cv2 as cv
import glob

# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
pattern_size = (5-1,5-1)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp *= 41
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
# 20,5 x 20,5 / 4,1 x 4,1

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Initialize webcam
cap = cv.VideoCapture(2)  # Change the index if you have multiple cameras

images = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting.")
        break

    images.append(frame)
    if len(images) > 25:
        images = images[1:]

    print(len(images))

    cv.imshow('frame', frame)

    sleep(2)

    # Break loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    #for i in range(0, len(images)):
        #cv.imwrite(filename=f"calibration/{i}.png", img=images[i])

for img in images:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(mtx)

cv.destroyAllWindows()
