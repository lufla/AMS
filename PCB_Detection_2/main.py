import numpy as np
import cv2 as cv
import math
import csv
import pandas as pd

gerber = cv.imread("./images/top_layer_image.png", cv.IMREAD_UNCHANGED)
gerber = cv.flip(gerber, 0)
gerber = cv.resize(gerber, (0,0), fx=0.05, fy=0.05, interpolation=cv.INTER_LINEAR)
print("gerber.shape: ", gerber.shape)

reference = cv.imread("./images/PP3_FPGA_Tester_Scan.png", cv.IMREAD_COLOR)
#reference = cv.resize(reference, (720, 480), interpolation=cv.INTER_LINEAR)
reference = cv.resize(reference, (0,0), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
print("reference.shape: ", reference.shape)

pnp_df = pd.read_csv("./PP3_FPGA_Tester/CAMOutputs/Assembly/PnP_PP3_FPGA_Tester_v3_front.txt",
    header=None, sep="\t", index_col=False, usecols=[0,1,2,3])
# 5:3, 160, 96, 160*90=14400
#pnp_df[1] = pnp_df[1] / 160
#pnp_df[2] = pnp_df[2] / 96
pnp_df[1] = pnp_df[1] * 90 * 0.05 / gerber.shape[1]
pnp_df[2] = pnp_df[2] * 90 * 0.05 / gerber.shape[0]
pnp_df[2] = 1 - pnp_df[2] # flip
print(pnp_df)


cap = cv.VideoCapture(0)
#cap.set(cv.CAP_PROP_FRAME_WIDTH, 720)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

moving_average_contour = np.float32([[[80,80],[640,80],[80,400],[640,400]]])

while True:
    e1 = cv.getTickCount()

    # Capture frame-by-frame
    ret, frame = cap.read()
 
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
        if area > 720*480*0.7:
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
    
    
    for i in range(0, pnp_df.shape[0]):
        pos = np.matmul(M_inv, [pnp_df.iloc[i][1]*gerber.shape[1], pnp_df.iloc[i][2]*gerber.shape[0], 1])[:2]
        cv.circle(frame, (int(pos[0]), int(pos[1])), 0, (255,255,127), 4)

    cv.imshow("reference", reference)
    cv.imshow('canny', canny)
    cv.imshow('frame', frame)
    cv.imshow('thresh', thresh)
    cv.imshow('perspective', perspective)
    cv.imshow('gerber', gerber)
    if cv.waitKey(1) == ord('q'):
        break

    e2 = cv.getTickCount()
    time = (e2 - e1) / cv.getTickFrequency()
    print(f'{time}s', end="\r", flush=True)
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# add filters?
# template matching
# history average statt moving average?
# remove outlier corner points?
# canny dilate and fill?
# cutout image at point
# find cutout in (perspective transformed / gripper cam) image