import os
import cv2
import numpy as np
import gerber
from gerber.render.cairo_backend import GerberCairoContext
import platform
from collections import deque

offset_x = 1
offset_y = 1

# Parameters for jitter reduction
frame_history = 5  # Number of frames to average
corner_history = deque(maxlen=frame_history)  # Store recent corner detections
min_area_threshold = 5000  # Minimum area to be considered as PCB


# Function to detect elements in Gerber files
def detect_top_layer_elements(input_files):
    traces, pads, holes, outlines, pcb_profile = [], [], [], [], []

    for file in input_files:
        gerber_file = gerber.read(file)
        for primitive in gerber_file.primitives:
            if isinstance(primitive, gerber.primitives.Line):
                if 'profile' in file.lower():
                    pcb_profile.append(primitive)
                else:
                    traces.append(primitive)
            elif isinstance(primitive, gerber.primitives.Circle):
                if primitive.hole_diameter > 0:
                    holes.append(primitive)
                else:
                    pads.append(primitive)
            elif isinstance(primitive, gerber.primitives.Rectangle) or isinstance(primitive, gerber.primitives.Obround):
                outlines.append(primitive)

    return traces, pads, holes, outlines, pcb_profile


# Function to create a transparent and vibrant image for the PCB overlay
def generate_vibrant_top_layer_image(outlines, pads, holes, pcb_profile, output_file='top_layer_image.png'):
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    for outline in outlines:
        x, y = outline.position
        width, height = outline.width, outline.height
        min_x, min_y = min(min_x, x - width / 2), min(min_y, y - height / 2)
        max_x, max_y = max(max_x, x + width / 2), max(max_y, y + height / 2)

    for line in pcb_profile:
        min_x = min(min_x, line.start[0], line.end[0])
        min_y = min(min_y, line.start[1], line.end[1])
        max_x = max(max_x, line.start[0], line.end[0])
        max_y = max(max_y, line.start[1], line.end[1])

    width = int((max_x - min_x) * 90)
    height = int((max_y - min_y) * 90)
    img = np.zeros((height, width, 4), dtype=np.uint8)

    for outline in outlines:
        top_left = (int((outline.position[0] - outline.width / 2 - min_x) * 90),
                    int((outline.position[1] - outline.height / 2 - min_y) * 90))
        bottom_right = (int((outline.position[0] + outline.width / 2 - min_x) * 90),
                        int((outline.position[1] + outline.height / 2 - min_y) * 90))
        color = (0, 255, 0, 200) if outline.width > 10 and outline.height > 10 else (0, 0, 255, 200)
        cv2.rectangle(img, top_left, bottom_right, color, -1)

    for pad in pads:
        center = (int((pad.position[0] - min_x) * 90), int((pad.position[1] - min_y) * 90))
        radius = int(pad.diameter / 4 * 90)
        cv2.circle(img, center, radius, (255, 0, 0, 200), -1)

    for hole in holes:
        center = (int((hole.position[0] - min_x) * 90), int((hole.position[1] - min_y) * 90))
        radius = int(hole.diameter / 4 * 90)
        cv2.circle(img, center, radius, (0, 255, 255, 200), -1)

    for line in pcb_profile:
        start = (int((line.start[0] - min_x) * 90), int((line.start[1] - min_y) * 90))
        end = (int((line.end[0] - min_x) * 90), int((line.end[1] - min_y) * 90))
        cv2.line(img, start, end, (255, 0, 255, 200), 3)
        cv2.circle(img, start, 5, (0, 255, 255, 255), -1)
        cv2.circle(img, end, 5, (0, 255, 255, 255), -1)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(output_file, img)


def average_corners(corners):
    if len(corner_history) < frame_history:
        return corners
    avg_corners = np.mean(corner_history, axis=0).astype(np.float32)
    return avg_corners


def show_overlay_with_webcam(overlay_image_path, calibration_data=None):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system() == 'Windows' else 0)
    if not cap.isOpened():
        raise IOError("Could not open webcam.")

    overlay = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
    overlay = cv2.flip(overlay, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if calibration_data:
            frame = cv2.undistort(frame, calibration_data['camera_matrix'], calibration_data['dist_coeffs'])

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Slightly expand the color range to increase the chance of detecting the lower right corner
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Morphological operations to reduce noise and shadows
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > min_area_threshold:
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

                if len(approx_corners) >= 4:
                    # Sort and average corners
                    approx_corners = sorted(approx_corners, key=lambda x: x[0][1])
                    top_points = sorted(approx_corners[:2], key=lambda x: x[0][0])
                    bottom_points = sorted(approx_corners[2:], key=lambda x: x[0][0])
                    src_points = np.float32(
                        [top_points[0][0], top_points[1][0], bottom_points[1][0], bottom_points[0][0]])

                    # Add to history and average corners
                    corner_history.append(src_points)
                    smoothed_corners = average_corners(src_points).astype(np.float32)

                    dst_points = np.float32([
                        [0, 0],
                        [overlay.shape[1], 0],
                        [overlay.shape[1], overlay.shape[0]],
                        [0, overlay.shape[0]]
                    ])

                    matrix = cv2.getPerspectiveTransform(dst_points, smoothed_corners)
                    transformed_overlay = cv2.warpPerspective(overlay, matrix, (frame.shape[1], frame.shape[0]))

                    overlay_alpha = transformed_overlay[:, :, 3] / 255.0
                    for c in range(0, 3):
                        frame[:, :, c] = (overlay_alpha * transformed_overlay[:, :, c] +
                                          (1 - overlay_alpha) * frame[:, :, c])

        larger_frame = cv2.resize(frame, None, fx=2, fy=2)
        cv2.imshow('Webcam with Larger Mirrored Overlay', larger_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Optional: Provide calibration data if available
# calibration_data = {'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs}

if __name__ == "__main__":
    gerber_directory = os.path.join(os.path.dirname(__file__), 'PP3_FPGA_Tester', 'CAMOutputs', 'GerberFiles')
    gerber_files = [
        os.path.join(gerber_directory, 'copper_top.gbr'),
        os.path.join(gerber_directory, 'soldermask_top.gbr'),
        os.path.join(gerber_directory, 'silkscreen_top.gbr'),
        os.path.join(gerber_directory, 'profile.gbr')
    ]

    missing_files = [file for file in gerber_files if not os.path.exists(file)]
    if missing_files:
        raise FileNotFoundError(f"One or more Gerber files were not found: {missing_files}")

    traces, pads, holes, outlines, pcb_profile = detect_top_layer_elements(gerber_files)
    filtered_holes = [hole for hole in holes if hole.diameter > 0.4]
    output_file = 'top_layer_image.png'
    generate_vibrant_top_layer_image(outlines, pads, filtered_holes, pcb_profile, output_file=output_file)
    show_overlay_with_webcam(output_file)
