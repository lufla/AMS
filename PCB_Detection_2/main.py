import numpy as np
import cv2 as cv
import math
import csv
import pandas as pd
import os
from dotenv import dotenv_values
import json

config = dotenv_values(".env")

with open(".env.json") as f:
    config_json = json.load(f)

K = np.float32(config_json["camera_matrix"])

K_inv = np.linalg.inv(K)

#file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PCB_Detection_2/images/top_layer_image.png")
gerber = cv.imread("PCB_Detection_2/images/top_layer_image.png", cv.IMREAD_UNCHANGED)

gerber = cv.flip(gerber, 0)
gerber = cv.resize(gerber, (0,0), fx=0.05, fy=0.05, interpolation=cv.INTER_LINEAR)
print("gerber.shape: ", gerber.shape)

reference = cv.imread("PCB_Detection_2/images/PP3_FPGA_Tester_Scan.png", cv.IMREAD_COLOR)
#reference = cv.resize(reference, (720, 480), interpolation=cv.INTER_LINEAR)
reference = cv.resize(reference, (0,0), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
print("reference.shape: ", reference.shape)

pnp_df = pd.read_csv("PCB_Detection_2/PP3_FPGA_Tester/CAMOutputs/Assembly/PnP_PP3_FPGA_Tester_v3_front.txt",
    header=None, sep="\t", index_col=False, usecols=[0,1,2,3])
# 5:3, 160, 96, 160*90=14400
#pnp_df[1] = pnp_df[1] / 160
#pnp_df[2] = pnp_df[2] / 96
# 90 pixel/mm, 0.05 resize
pnp_df[1] = pnp_df[1] * 90 * 0.05 / gerber.shape[1]
pnp_df[2] = pnp_df[2] * 90 * 0.05 / gerber.shape[0]
pnp_df[2] = 1 - pnp_df[2] # flip
print(pnp_df)

gerber_size_mm = np.array(gerber.shape) / 0.05 / 90

# IC1 coordinates
ic1_center_pixel = (
    int(pnp_df.iloc[25][1] * reference.shape[1]),
    int(pnp_df.iloc[25][2] * reference.shape[0])
)
cutout_radius = 50


cap = cv.VideoCapture(int(config["camera_index"]))
#cap.set(cv.CAP_PROP_FRAME_WIDTH, 720)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

moving_average_contour = np.float32([[[80,80],[640,80],[80,400],[640,400]]])
tvecs_history = []

while True:
    e1 = cv.getTickCount()

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_copy = frame.copy()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #cv.circle(frame, (10, 10), 0, (0,0,255), 2)

    #ret, thresh = cv.threshold(gray, 127, 255, 0)
    #thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,101,-2)
    gray_blur = cv.GaussianBlur(gray,(5,5),0)
    ret,thresh = cv.threshold(gray_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    thresh = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=2)

    canny = cv.Canny(frame,100,200)

    contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #cv.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0))

    for i in range(0, len(contours)):
        area = cv.contourArea(contours[i])
        if area > 200:
            #cv.drawContours(image=frame, contours=contours[i], contourIdx=-1, color=(0, 255, 0), thickness=3)
            pass
    
    if contours:
        cnt = contours[0]
        epsilon = 0.1*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,epsilon,True)
        #cv.drawContours(image=frame, contours=approx, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv.LINE_AA)
        
    
    contour_list = []
    hull_list = []
    approx_list = []
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < 200:
            continue
        if area > frame.shape[0]*frame.shape[1]*0.9:
            continue

        contour_list.append(contours[i])

        hull = cv.convexHull(contours[i])
        hull_list.append(hull)

        epsilon = 0.1*cv.arcLength(contours[i],True)
        #approx = cv.approxPolyDP(curve=contours[i],epsilon=epsilon,closed=True)
        approx = cv.approxPolyDP(curve=hull,epsilon=epsilon,closed=True)
        #approx = cv.approxPolyN(curve=contours[i],nsides=4,epsilon_percentage=-1.0,ensure_convex=True) #opencv 4.10 dev
        approx_list.append(approx)
    
    cv.drawContours(image=frame, contours=contour_list, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    cv.drawContours(image=frame, contours=hull_list, contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.drawContours(image=frame, contours=approx_list, contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv.LINE_AA)

    # get largest area approximation with 4 corners
    approx_list_filtered = []
    for i in range(len(approx_list)):
        if len(approx_list[i]) != 4:
            continue
        if cv.contourArea(approx_list[i]) < 720*480*0.8/32:
            continue
        
        approx_list_filtered.append(approx_list[i])

    approx_list_filtered.sort(key=cv.contourArea, reverse=True)

    cv.drawContours(image=frame, contours=approx_list_filtered, contourIdx=-1, color=(127, 127, 255), thickness=4, lineType=cv.LINE_AA)

    if approx_list_filtered:
        moving_average_contour = np.add(moving_average_contour*0.8, np.reshape(approx_list_filtered[0]*0.2, (1,4,2)))
    
    pts1 = np.float32(moving_average_contour[0][[0,1,3,2]])
    pts2 = np.float32([[0,0],[gerber.shape[1],0],[0,gerber.shape[0]],[gerber.shape[1],gerber.shape[0]]])
    
    M = cv.getPerspectiveTransform(pts1,pts2)
    perspective = cv.warpPerspective(frame,M,(gerber.shape[1], gerber.shape[0]))
    perspective_clean = cv.warpPerspective(frame_copy,M,(gerber.shape[1], gerber.shape[0]))

    # get ic cutout
    cutout_x_min = max(ic1_center_pixel[0] - cutout_radius, 0)
    cutout_x_max = min(ic1_center_pixel[0] + cutout_radius, reference.shape[1] - 1)
    cutout_y_min = max(ic1_center_pixel[1] - cutout_radius, 0)
    cutout_y_max = min(ic1_center_pixel[1] + cutout_radius, reference.shape[0] - 1)
    ic_cutout_center_pixel = (ic1_center_pixel[0] - cutout_x_min, ic1_center_pixel[1] - cutout_y_min)
    
    reference_cutout = reference[cutout_y_min:cutout_y_max, cutout_x_min:cutout_x_max]
    reference_cutout_canny = cv.Canny(reference_cutout,100,200)
    reference_cutout_gray = cv.cvtColor(reference_cutout, cv.COLOR_BGR2GRAY)

    input_cutout = frame_copy[:, :] # placeholder for gripper cam
    input_cutout_canny = cv.Canny(input_cutout,100,200)
    input_cutout_gray = cv.cvtColor(input_cutout, cv.COLOR_BGR2GRAY)

    # Apply template Matching
    method = cv.TM_CCOEFF_NORMED
    res = cv.matchTemplate(input_cutout_gray, reference_cutout_gray, method)
    # NORMED method works better for different lighting
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + reference_cutout.shape[1], top_left[1] + reference_cutout.shape[0])
    cv.rectangle(input_cutout,top_left, bottom_right, 255, 2)
    # TODO average cutout position?


    """
    lines = cv.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(frame, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    """

    """
    linesP = cv.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    """

    for i in range(0, pnp_df.shape[0]):
        pos = (int(pnp_df.iloc[i][1]*gerber.shape[1]), int(pnp_df.iloc[i][2]*gerber.shape[0]))
        cv.circle(gerber, (int(pos[0]), int(pos[1])), 0, (255,255,127), 4)
        cv.circle(perspective, (int(pos[0]), int(pos[1])), 0, (255,255,127), 4)

    dst_points = np.float32([
        [0, 0],
        [gerber.shape[1], 0],
        [0, gerber.shape[0]],
        [gerber.shape[1], gerber.shape[0]],
    ])
    
    M_inv = cv.getPerspectiveTransform(dst_points, pts1)
    transformed_overlay = cv.warpPerspective(gerber, M_inv, (frame.shape[1], frame.shape[0]))

    overlay_alpha = transformed_overlay[:, :, 3] / 255.0
    for c in range(0, 3):
        frame[:, :, c] = (overlay_alpha * transformed_overlay[:, :, c] +
                            (1 - overlay_alpha) * frame[:, :, c])
    
    overlay_alpha = gerber[:, :, 3] / 255.0
    for c in range(0, 3):
        perspective[:, :, c] = (overlay_alpha * gerber[:, :, c] +
                            (1 - overlay_alpha) * perspective[:, :, c])

    pcb_points = []
    camera_screen_points = []

    for i in range(0, pnp_df.shape[0]):
        pos_pcb_screen = [pnp_df.iloc[i][1]*gerber.shape[1], pnp_df.iloc[i][2]*gerber.shape[0], 1]
        pos_camera_screen = np.matmul(M_inv, pos_pcb_screen)
        pos_camera_screen_norm = pos_camera_screen / pos_camera_screen[2]

        #
        pcb_points.append(np.array([pnp_df.iloc[i][1]*gerber_size_mm[1], pnp_df.iloc[i][2]*gerber_size_mm[0]]))
        camera_screen_points.append(pos_camera_screen_norm)

        cv.circle(frame, (int(pos_camera_screen_norm[0]), int(pos_camera_screen_norm[1])), 0, (255,255,127), 4)

        cv.circle(frame, (int(pos_camera_screen[0]), int(pos_camera_screen[1])), 0, (255,0,255), 4)

        if i == 25:
            pos_3d = np.matmul(K_inv, pos_camera_screen_norm)
            cv.circle(frame, np.int32(pos_camera_screen_norm[:2]), 4, (0,255,0), 4)

            #print(pos_3d)
            #print(": ", M[2,:])

        # test, funktioniert nur wenn kamera gerade über objekt ist
        #distance_estimate = np.sqrt(np.linalg.det(M[0:2, 0:2]) * 0.8) * 140
        #print("distance_estimate: ", distance_estimate)

    obj_points = [np.float32([[x[0], x[1], 0] for x in pcb_points[0:]])]
    imgpoints = [np.float32([[x[0:2]] for x in camera_screen_points[0:]])]

    obj_points_pnp = np.float32([[x[0], x[1], 0] for x in pcb_points])
    imgpoints_pnp = np.float32([x[0:2] for x in camera_screen_points])
    
    #obj_points = [np.float32([
    #    [0, 0, 0],
    #    [gerber_size_mm[1], 0, 0],
    #    [0, gerber_size_mm[0], 0],
    #    [gerber_size_mm[1], gerber_size_mm[0], 0],
    #])]
    #imgpoints = [np.float32([[moving_average_contour[0][0]], [moving_average_contour[0][1]], [moving_average_contour[0][2]], [moving_average_contour[0][3]]])]
    
    #print(obj_points_pnp.shape)
    #print(imgpoints_pnp.shape)
    #ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, imgpoints, gray.shape[::-1], None, None)
    ret, rvecs, tvecs = cv.solvePnP(obj_points_pnp, imgpoints_pnp, K, None)
    #print(rvecs)
    #print(tvecs)
    #print()
    tvecs_history.append(np.reshape(tvecs, (3)))
    if len(tvecs_history) > 60:
        tvecs_history = tvecs_history[1:]
    tvecs_average = np.average(tvecs, 0)
    #print(tvecs_average)
    #print()
    ic1_position = np.float32([
            pnp_df.iloc[25][1] * gerber.shape[1] / 0.05 / 90,
            pnp_df.iloc[25][1] * gerber.shape[0] / 0.05 / 90,
            0
        ]).reshape(3,1)
    rmat = cv.Rodrigues(rvecs)[0]
    #print(rmat)
    ic1_camera_point = (np.matmul(rmat, ic1_position) + tvecs).reshape(3)
    print(ic1_camera_point)

    axis = np.float32([[50,0,0], [0,50,0], [0,0,-50]]).reshape(-1,3)
    axis_image, jac = cv.projectPoints(axis, rvecs, tvecs, K, None)
    axis_image = np.int32(axis_image)
    #print(axis_image)
    corner = np.matmul(M_inv, [0, 0, 1])
    corner = corner / corner[2]
    corner = np.int32(corner[0:2])
    #print(corner)
    frame = cv.line(frame, corner, axis_image[0][0], (255,0,0), 5)
    frame = cv.line(frame, corner, axis_image[1][0], (0,255,0), 5)
    frame = cv.line(frame, corner, axis_image[2][0], (0,0,255), 5)

    cv.imshow("reference", reference)
    cv.imshow('canny', canny)
    cv.imshow('frame', frame)
    cv.imshow('thresh', thresh)
    cv.imshow('perspective', perspective)
    cv.imshow('gerber', gerber)
    cv.imshow('input_cutout', input_cutout)
    cv.imshow('reference_cutout', reference_cutout)
    if cv.waitKey(1) == ord('q'):
        break

    e2 = cv.getTickCount()
    time = (e2 - e1) / cv.getTickFrequency()
    print(f'{time}s', end="\r", flush=True)
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# add filters?
# history average statt moving average?
# remove outlier corner points?
# canny dilate and fill?

# detect contour with green colour
# TODO switch to gripper cam when template detection threshold is reached