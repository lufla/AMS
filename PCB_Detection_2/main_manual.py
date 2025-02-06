import base64
import json
import os
import re
import math
import csv
import time
import cv2
import cv2 as cv
import numpy as np
import pandas as pd
#import easyocr
from dotenv import load_dotenv, dotenv_values
import roslibpy
import scipy
import scipy.spatial

load_dotenv()  # Load env variables (confidenceThreshold, etc.)

CAMERA_WEBCAM = 1
CAMERA_HEAD = 2
CAMERA_GRIPPER = 3

USE_WEBCAM = False

confidence_threshold = os.getenv("confidenceThreshold")
if confidence_threshold is None:
    confidence_threshold = 0.95
else:
    try:
        confidence_threshold = float(confidence_threshold)
    except ValueError:
        print("Invalid confidenceThreshold. Using default 0.95.")
        confidence_threshold = 0.95
print(f"Confidence Threshold: {confidence_threshold}")

tiago_image_head_cache = None
tiago_image_gripper_cache = None
tiago_head_arm_transform_cache = None

def calculate_confidence(contour, mask, frame):
    confidence = 1
    area = cv2.contourArea(contour)
    if area < 400:
        return 0.0
    elif area > 1000:
        confidence += 0.2

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if 0.9 <= aspect_ratio <= 1.1:
        confidence += 0.2
    else:
        confidence -= 0.3

    # Check if it's dark inside
    mask_fill = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(mask_fill, [contour], -1, 255, -1)
    mean_color = cv2.mean(frame, mask=mask_fill)[:3]
    if np.mean(mean_color) < 50:
        confidence += 0.2
    else:
        confidence -= 0.3

    return max(0.0, min(1.0, confidence))

def detect_black_squares(frame):
    """Detect black squares in 'frame' and draw bounding boxes for confident detections."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_ics = []

    for contour in contours:
        conf = calculate_confidence(contour, mask, frame)
        if conf > confidence_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            detected_ics.append(((x, y, w, h), conf))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"IC ({conf:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    return frame, detected_ics

def map_text_to_ic(frame, detected_ics, results):
    """Map recognized text (results) to the nearest black-square bounding box."""
    for (bbox, text, conf) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

        # Draw bounding box for recognized text
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Find closest detected IC
        text_center = ((top_left[0] + bottom_right[0]) // 2,
                       (top_left[1] + bottom_right[1]) // 2)
        closest_ic = None
        min_distance = float('inf')

        for (ic_bbox, _) in detected_ics:
            x, y, w, h = ic_bbox
            ic_center = (x + w // 2, y + h // 2)
            dist = np.sqrt((text_center[0] - ic_center[0]) ** 2 + (text_center[1] - ic_center[1]) ** 2)
            if dist < min_distance:
                min_distance = dist
                closest_ic = ic_bbox

        if closest_ic:
            x, y, w, h = closest_ic
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def process_frame_with_rotations_in_one_window(frame, reader, pattern):
    """
    Rotate the input 'frame' by 0, 90, 180, 270 degrees.
    Detect black squares + OCR text on each rotation.
    Then combine all four rotated views into a single bigger window
    called 'All Rotations'.
    """
    rotations = [0, 90, 180, 270]
    rotated_frames = []

    for angle in rotations:
        # ----- 1) Rotate -----
        if angle == 0:
            rot_frame = frame.copy()
        else:
            rotation_flag = (
                cv2.ROTATE_90_CLOCKWISE if angle == 90 else
                cv2.ROTATE_180         if angle == 180 else
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )
            rot_frame = cv2.rotate(frame, rotation_flag)

        # ----- 2) Detect black squares on this rotated frame -----
        rot_frame, detected_ics = detect_black_squares(rot_frame)

        # ----- 3) OCR text detection -----
        gray = cv2.cvtColor(rot_frame, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=1)

        # ----- 4) Filter results to match "IC\d+" pattern -----
        filtered_results = [
            (bbox, text, c)
            for (bbox, text, c) in results
            if pattern.match(text)
        ]

        # ----- 5) Map recognized text to the closest detected IC -----
        map_text_to_ic(rot_frame, detected_ics, filtered_results)

        rotated_frames.append(rot_frame)

    # Now we have 4 rotated frames in rotated_frames.
    # We'll unify their sizes and display them in a 2×2 grid in one window.

    def unify_size(img_list):
        # Find the largest width and height among all frames
        heights = [img.shape[0] for img in img_list]
        widths  = [img.shape[1] for img in img_list]
        max_h   = max(heights)
        max_w   = max(widths)

        # Resize each image to (max_w, max_h)
        resized = []
        for im in img_list:
            resized.append(cv2.resize(im, (max_w, max_h)))
        return resized

    # Resize all frames to the same dimensions
    rotated_frames = unify_size(rotated_frames)

    # Build a 2×2 grid: top_row for 0°, 90°, bottom_row for 180°, 270°
    top_row    = np.hstack([rotated_frames[0], rotated_frames[1]])
    bottom_row = np.hstack([rotated_frames[2], rotated_frames[3]])
    combined_image = np.vstack([top_row, bottom_row])

    # Show the combined image in one window
    cv2.imshow("All Rotations", combined_image)


def load_config(env_path=".env"):
    return dotenv_values(env_path)

def load_intrinsic_matrix(CAMERA):

    with open(".env.json") as f:
        config_json = json.load(f)

    if CAMERA == CAMERA_WEBCAM: K = np.float32(config_json["camera_matrix"])
    if CAMERA == CAMERA_HEAD: K = np.float32(config_json["camera_head_matrix"])
    if CAMERA == CAMERA_GRIPPER: K = np.float32(config_json["camera_gripper_matrix"])

    K_inv = np.linalg.inv(K)
    return K, K_inv

def load_gerber_image(image_path="PCB_Detection_2/images/top_layer_image.png", scale_factor=0.05):
    gerber = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if gerber is None:
        raise FileNotFoundError(f"Gerber image not found at {image_path}")
    gerber = cv.flip(gerber, 0)
    gerber = cv.resize(gerber, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
    return gerber

def load_reference_image(image_path="PCB_Detection_2/images/PP3_FPGA_Tester_Scan.png", scale_factor=0.25):
    reference = cv.imread(image_path, cv.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(f"Reference image not found at {image_path}")
    reference = cv.resize(reference, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
    return reference

def load_pnp_data(csv_path, gerber_shape=(0, 0), scale_x=90 * 0.05, scale_y=90 * 0.05):
    pnp_df = pd.read_csv(csv_path, header=None, sep="\t", index_col=False, usecols=[0, 1, 2, 3])
    if gerber_shape[1] == 0 or gerber_shape[0] == 0:
        raise ValueError("Invalid gerber_shape provided for PnP data scaling.")

    pnp_df[1] = pnp_df[1] * scale_x / gerber_shape[1]
    pnp_df[2] = pnp_df[2] * scale_y / gerber_shape[0]
    pnp_df[2] = 1 - pnp_df[2]  # Flip Y-axis
    return pnp_df

def calculate_ic_center(pnp_df, reference_shape, index=25):
    ic_center_pixel = (
        int(pnp_df.iloc[index][1] * reference_shape[1]),
        int(pnp_df.iloc[index][2] * reference_shape[0])
    )
    cutout_radius = 50
    return ic_center_pixel, cutout_radius

def decode_image_message(message):
    base64_bytes = message['data'].encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv.imdecode(jpg_as_np, cv.IMREAD_COLOR)
    return image

def initialize_ros_connection():
    client = roslibpy.Ros(host=os.getenv("ros_host"), port=9090)
    client.run()
    return client

def initialize_tiago_head_camera(client):
    topic = roslibpy.Topic(client, '/xtion/rgb/image_raw/compressed', 'sensor_msgs/CompressedImage')
    def set_head_cache(img):
        global tiago_image_head_cache
        tiago_image_head_cache = img
    topic.subscribe(lambda message: set_head_cache(decode_image_message(message)))

def initialize_tiago_gripper_camera(client):
    topic = roslibpy.Topic(client, '/end_effector_camera/image_raw/compressed', 'sensor_msgs/CompressedImage')
    def set_gripper_cache(img):
        global tiago_image_gripper_cache
        tiago_image_gripper_cache = img
    topic.subscribe(lambda message: set_gripper_cache(decode_image_message(message)))

tf_stream_topic = None
def initialize_head_arm_transform_stream(client):

    topic_result = roslibpy.Topic(client, '/tf_stream/result', 'tf_lookup/TfStreamActionResult')

    def check_tf_stream_result(message):
        global tf_stream_topic
        tf_stream_topic = message["result"]["topic"]

    topic_result.subscribe(lambda message: check_tf_stream_result(message))

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
        print("Waiting for transform stream topic")
        time.sleep(1)

    topic_stream = roslibpy.Topic(client, tf_stream_topic, 'tf/tfMessage')

    def set_head_arm_transform_cache(message):
        global tiago_head_arm_transform_cache
        translation = message["transforms"][0]["transform"]["translation"]
        tvec = [translation["x"], translation["y"], translation["z"]]
        rot_q = message["transforms"][0]["transform"]["rotation"]
        rot_q_array = [rot_q["x"], rot_q["y"], rot_q["z"], rot_q["w"]]
        rotation = scipy.spatial.transform.Rotation.from_quat(rot_q_array)
        rmat = rotation.as_matrix()
        tiago_head_arm_transform_cache = {"tvec": tvec, "rmat": rmat}



    topic_stream.subscribe(lambda message: set_head_arm_transform_cache(message))

def initialize_camera(camera_index=0, width=None, height=None):
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("Cannot open camera index:", camera_index)

    if width and height:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    return cap, actual_width, actual_height

def preprocess_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(gray_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresh = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=2)
    canny = cv.Canny(frame, 100, 200)
    return gray, thresh, canny

def find_and_draw_contours(frame, thresh):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_list, hull_list, approx_list = [], [], []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if 200 < area < frame.shape[0] * frame.shape[1] * 0.2: # min and max contour area
            contour_list.append(cnt)
            hull = cv.convexHull(cnt)
            hull_list.append(hull)
            epsilon = 0.1 * cv.arcLength(hull, True)
            approx = cv.approxPolyDP(hull, epsilon, True)
            approx_list.append(approx)

    # Draw them
    cv.drawContours(frame, contour_list, -1, (0, 255, 0), 2, cv.LINE_AA)
    cv.drawContours(frame, hull_list, -1, (255, 0, 0), 2, cv.LINE_AA)
    cv.drawContours(frame, approx_list, -1, (0, 0, 255), 2, cv.LINE_AA)
    return approx_list

def filter_approximations(approx_list, frame_shape, min_area_ratio=0.8 / 32):
    filtered = [
        approx for approx in approx_list
        if len(approx) == 4 and
           cv.contourArea(approx) >= frame_shape[0] * frame_shape[1] * min_area_ratio
    ]
    filtered.sort(key=cv.contourArea, reverse=True)
    return filtered

def update_moving_average_contour(moving_avg, new_contour, alpha=0.8):
    if new_contour is not None:
        moving_avg = alpha * moving_avg + (1 - alpha) * new_contour.reshape(1, 4, 2).astype(np.float32)
    return moving_avg

def compute_perspective_transform(moving_avg, gerber_shape, frame_shape, frame):
    pts1 = np.float32(moving_avg[0][[0,1,3,2]])
    pts2 = np.float32([
        [0, 0],
        [gerber_shape[1], 0],
        [0, gerber_shape[0]],
        [gerber_shape[1], gerber_shape[0]]
    ])
    M = cv.getPerspectiveTransform(pts1, pts2)
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = None
        print("Error: cannot invert perspective matrix.")

    perspective = cv.warpPerspective(frame, M, (gerber_shape[1], gerber_shape[0]))
    perspective_clean = cv.warpPerspective(frame.copy(), M, (gerber_shape[1], gerber_shape[0]))
    return M, M_inv, perspective, perspective_clean

def get_ic_cutout(reference, ic_center, radius):
    x_min = max(ic_center[0] - radius, 0)
    x_max = min(ic_center[0] + radius, reference.shape[1] - 1)
    y_min = max(ic_center[1] - radius, 0)
    y_max = min(ic_center[1] + radius, reference.shape[0] - 1)

    reference_cutout = reference[y_min:y_max, x_min:x_max]
    reference_cutout_canny = cv.Canny(reference_cutout, 100, 200)
    reference_cutout_gray = cv.cvtColor(reference_cutout, cv.COLOR_BGR2GRAY)
    ic_cutout_center = (ic_center[0] - x_min, ic_center[1] - y_min)
    return reference_cutout, reference_cutout_canny, reference_cutout_gray, ic_cutout_center

def perform_template_matching(input_cutout_gray, reference_cutout_gray, method=cv.TM_CCOEFF_NORMED):
    res = cv.matchTemplate(input_cutout_gray, reference_cutout_gray, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (
        top_left[0] + reference_cutout_gray.shape[1],
        top_left[1] + reference_cutout_gray.shape[0]
    )
    return top_left, bottom_right

def overlay_images(frame, transformed_overlay, alpha_channel_index=3):
    if transformed_overlay.shape[2] <= alpha_channel_index:
        return frame
    overlay_alpha = transformed_overlay[:, :, alpha_channel_index] / 255.0
    for c in range(3):
        frame[:, :, c] = (
            overlay_alpha * transformed_overlay[:, :, c]
            + (1 - overlay_alpha) * frame[:, :, c]
        )
    return frame

def calibrate_camera(obj_points, img_points, image_shape):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, image_shape, None, None)
    return ret, mtx, dist, rvecs, tvecs

def draw_pcb_points(gerber, perspective, pnp_df, gerber_size_mm, M_inv, frame, K_inv):
    pcb_points = []
    camera_screen_points = []

    for i in range(pnp_df.shape[0]):
        pos_pcb_screen = [
            pnp_df.iloc[i][1] * gerber.shape[1],
            pnp_df.iloc[i][2] * gerber.shape[0],
            1
        ]
        # Transform from gerber space -> camera space
        if M_inv is not None:
            pos_camera_screen = np.matmul(M_inv, pos_pcb_screen)
            pos_camera_screen_norm = pos_camera_screen / pos_camera_screen[2]
        else:
            pos_camera_screen_norm = np.array([0, 0, 1])

        # Save for calibration
        pcb_points.append(np.array([
            pnp_df.iloc[i][1] * gerber_size_mm[1],
            pnp_df.iloc[i][2] * gerber_size_mm[0]
        ]))
        camera_screen_points.append(pos_camera_screen_norm)

        # Draw circles
        if M_inv is not None:
            cv.circle(frame,
                      (int(pos_camera_screen_norm[0]), int(pos_camera_screen_norm[1])),
                      0, (255, 255, 127), 4)
            cv.circle(frame,
                      (int(pos_camera_screen[0]), int(pos_camera_screen[1])),
                      0, (255, 0, 255), 4)

        # Example usage
        if i == 25:
            pos_3d = np.matmul(K_inv, pos_camera_screen_norm)
            # Additional logic if needed

    return pcb_points, camera_screen_points

def setup(states):
    """
    A single main() that runs:
      - Black squares detection + OCR (in 4 rotations shown in one window)
      - Calibration + Gerber overlay
    on the same camera feed and frames.
    """

    # (A) SETUP FOR PART 1 (BLACK SQUARE + OCR)
    #states["reader"] = easyocr.Reader(['en'], gpu=True)
    #states["pattern"] = re.compile(r'^IC\d+$')

    # (B) SETUP FOR PART 2 (CALIBRATION + OVERLAY)
    config = load_config()
    camera_index = int(config.get("camera_index", 0))
    if USE_WEBCAM:
        states["K"], states["K_inv"] = load_intrinsic_matrix(CAMERA_WEBCAM)
    else:
        states["K_head"], states["K_head_inv"] = load_intrinsic_matrix(CAMERA_HEAD)
        states["K_gripper"], states["K_gripper_inv"] = load_intrinsic_matrix(CAMERA_GRIPPER)

    states["gerber"] = load_gerber_image("PCB_Detection_2/images/top_layer_image.png", scale_factor=0.05)
    states["reference"] = load_reference_image("PCB_Detection_2/images/PP3_FPGA_Tester_Scan.png", scale_factor=0.25)
    states["gerber_size_mm"] = np.array(states["gerber"].shape) / 0.05 / 90

    states["pnp_df"] = load_pnp_data(
        csv_path="PCB_Detection_2/PP3_FPGA_Tester/CAMOutputs/Assembly/PnP_PP3_FPGA_Tester_v3_front.txt",
        gerber_shape=states["gerber"].shape
    )

    if USE_WEBCAM:
        cap, width, height = initialize_camera(camera_index)
        print(f"Camera opened with width={width}, height={height}")

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
        states["cap"] = cap
    else:
        client = initialize_ros_connection()
        states["client"] = client
        initialize_tiago_head_camera(client)
        initialize_tiago_gripper_camera(client)
        initialize_head_arm_transform_stream(client)

        while(
            tiago_image_head_cache is None
            or tiago_image_gripper_cache is None
            or tiago_head_arm_transform_cache):
            if tiago_image_head_cache is None:
                print("Waiting for head camera images")
            if tiago_image_gripper_cache is None:
                print("Waiting for gripper camera images")
            if tiago_head_arm_transform_cache is None:
                print("Waiting for head arm transform")
            time.sleep(1)


def printhello(states):
    print("hello")

def detect_cutout(states, component_index):
    print("Press 'q' to quit.")
    while True:
        # Grab frame
        if USE_WEBCAM:
            ret, frame = states["cap"].read()
 
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            K = states["K"]
            K_inv = states["K_inv"]
        else:
            frame = tiago_image_gripper_cache.copy()
        
            if frame is None:
                print("Failed to read frame. Exiting.")
                break
            
            K = states["K_gripper"]
            K_inv = states["K_gripper_inv"]

        # 5) Template matching for a known IC
        ic_center_pixel, cutout_radius = calculate_ic_center(states["pnp_df"], states["reference"].shape, component_index)
        reference_cutout, ref_cutout_canny, ref_cutout_gray, ic_cutout_center = get_ic_cutout(
            states["reference"], ic_center_pixel, cutout_radius
        )
        input_cutout_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        try:
            top_left, bottom_right = perform_template_matching(input_cutout_gray, ref_cutout_gray)
            cv.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        except cv.error as e:
            print(f"Template matching error: {e}")

def detect_component(states):
    print("Press 'q' to quit.")
    while True:
        # Grab frame
        if USE_WEBCAM:
            ret, frame = states["cap"].read()
 
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            K = states["K"]
            K_inv = states["K_inv"]
        else:
            frame = tiago_image_gripper_cache.copy()
        
            if frame is None:
                print("Failed to read frame. Exiting.")
                break
            
            K = states["K_gripper"]
            K_inv = states["K_gripper_inv"]

        # Show all four rotations in "All Rotations" window
        process_frame_with_rotations_in_one_window(frame, states["reader"], states["pattern"])

    pass

def detect_pcb(states):
    # For perspective tracking
    moving_average_contour = np.float32([[[80,80], [640,80], [80,400], [640,400]]])

    print("Press 'q' to quit.")
    while True:
        e1 = cv.getTickCount()

        # Grab frame
        if USE_WEBCAM:
            ret, frame = states["cap"].read()
 
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            K = states["K"]
            K_inv = states["K_inv"]
        else:
            frame = tiago_image_head_cache.copy()
        
            if frame is None:
                print("Failed to read frame. Exiting.")
                break

            K = states["K_head"]
            K_inv = states["K_head_inv"]

        frame_clean = frame.copy()

        # 1) Preprocessing
        gray, thresh, canny = preprocess_frame(frame)

        # 2) Find major quadrilaterals
        approx_list = find_and_draw_contours(frame, thresh)
        approx_filtered = filter_approximations(approx_list, frame.shape)
        if approx_filtered:
            cv.drawContours(frame, approx_filtered, -1, (127, 127, 255), 4, cv.LINE_AA)

            """
            # find average PCB Color
            mask = np.zeros(frame_clean.shape, np.uint8)
            print(approx_filtered[0][0][0])
            print(approx_filtered[1][0][0])
            print(approx_filtered[2][0][0])
            print(approx_filtered[3][0][0])
            mask = cv.drawContours(mask, approx_filtered, -1, (255,255,255),1)
            average_color = cv.mean(frame_clean, mask)
            print("avg col: ", average_color)
            """

        # 3) Update moving average contour
        if approx_filtered:
            moving_average_contour = update_moving_average_contour(moving_average_contour, approx_filtered[0])

        # 4) Perspective transform
        try:
            M, M_inv, perspective, perspective_clean = compute_perspective_transform(
                moving_average_contour, states["gerber"].shape, frame.shape, frame
            )

            
        except cv.error as e:
            print(f"Perspective transform error: {e}")
            continue

        # 6) Draw PnP-based PCB points
        pcb_points, camera_screen_points = draw_pcb_points(
            states["gerber"], perspective, states["pnp_df"], states["gerber_size_mm"], M_inv, frame, K_inv
        )

        # 7) Overlay Gerber image onto the live camera
        try:
            if M_inv is not None:
                transformed_overlay = cv.warpPerspective(states["gerber"], M_inv, (frame.shape[1], frame.shape[0]))
                frame = overlay_images(frame, transformed_overlay)
        except cv.error as e:
            print(f"Overlay error: {e}")

        # 8) Camera calibration (if enough points)
        obj_points_pnp = np.float32([[x[0], x[1], 0] for x in pcb_points])
        imgpoints_pnp = np.float32([x[0:2] for x in camera_screen_points])

        ret, rvecs, tvecs = cv.solvePnP(obj_points_pnp, imgpoints_pnp, K, None)
        states["pcb_rvecs"] = rvecs
        states["pcb_tvecs"] = tvecs

        pcb_corner = np.float32([0, 0, 0]).reshape(3,1)

        rmat = cv.Rodrigues(rvecs)[0]
        #print(rmat)
        pcb_corner_camera_point = (np.matmul(rmat, pcb_corner) + tvecs).reshape(3)
        print(pcb_corner_camera_point)

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

        # 9) Show windows from Part 2
        cv.imshow("Frame", frame)
        #cv.imshow("Canny", canny)
        #cv.imshow("Threshold", thresh)
        #cv.imshow("Perspective", perspective)
        cv.imshow("Perspective Clean", perspective_clean)
        #cv.imshow("Gerber", states["gerber"])
        #cv.imshow("Gerber", states["gerber"])

        #cv.imshow("Input Cutout", frame_copy)  # Show the rectangle for template matching
        #cv.imshow("Reference Cutout", reference_cutout)

        # Quit check
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        e2 = cv.getTickCount()
        elapsed_time = (e2 - e1) / cv.getTickFrequency()
        #print(f"Processing Time: {elapsed_time:.6f}s", end="\r", flush=True)

    #cap.release()
    cv.destroyAllWindows()

def calculate_component_pcb_position(states, component_index = 25):
    # check states
    if states["pcb_rvecs"] is None:
        print("Error: \"pcb_rvecs\" is None")
        return
    
    if states["pcb_tvecs"] is None:
        print("Error: \"pcb_tvecs\" is None")
        return
    
    ic_center_pixel, cutout_radius = calculate_ic_center(states["pnp_df"], states["reference"].shape, component_index)

    ic1_position = np.float32([
        states["pnp_df"].iloc[25][1] * states["gerber"].shape[1] / 0.05 / 90,
        states["pnp_df"].iloc[25][1] * states["gerber"].shape[0] / 0.05 / 90,
        0
    ]).reshape(3,1)
    rmat = cv.Rodrigues(states["pcb_rvecs"])[0]
    ic1_camera_point = (np.matmul(rmat, ic1_position) + states["pcb_tvecs"]).reshape(3)
    states["component_position_camera"] = ic1_camera_point

    ic1_camera_point_yzflip = np.float32([ic1_camera_point[0],ic1_camera_point[2], ic1_camera_point[1]])
    states["component_position_arm"] = (
        np.matmul(tiago_head_arm_transform_cache["rmat"], ic1_camera_point_yzflip/1000) + tiago_head_arm_transform_cache["tvec"]
    ).reshape(3)
    print("camera: ", states["component_position_camera"])
    print("arm: ", states["component_position_arm"])
    

def set_component_index(states):
    component_index = input("Enter component_index:")
    states["component_index"] = component_index
    print(f"Selected component {component_index}: {states['pnp_df'].iloc[component_index][0]}")


def move_arm_over_component(states):
    topic = roslibpy.Topic(states["client"], '/thk_ns/thk_tiago_xya', 'std_msgs/String')

    target = {"x": states["component_position_arm"][1], "y": -states["component_position_arm"][0], "angle": math.pi/2}
    print("move to: ", target)

    topic.publish(target)

def move_head(states):
    topic = roslibpy.Topic(states["client"], '/head_controller/command', 'std_msgs/String')

    topic.publish(
        {
            "joint_names":  ["head_1_joint", "head_2_joint"],
            "points": [{"positions": [0.0, -0.85], "time_from_start": {"secs": 1}}]
        }
    )

def reset_arm(states):
    topic = roslibpy.Topic(states["client"], '/thk_ns/thk_tiago_xya', 'std_msgs/String')

    topic.publish({"x": 0.1, "y": 0.95, "angle": math.pi/2})

def show_head_transform(states):
    print(tiago_head_arm_transform_cache)
    print()


# State
states = {
    "client": None,

    "K": None,
    "K_inv": None,
    "K_head": None,
    "K_head_inv": None,
    "K_gripper": None,
    "K_gripper_inv": None,

    "gerber": None,
    "gerber_size_mm": None,
    "reference": None,
    "pnp_df": None,

    "pcb_rvecs": None,
    "pcb_tvecs": None,
    "component_index": None,
    "component_position_camera": None,
    "component_position_arm": None,
}

def end(states):
    states["client"].terminate()
    exit()

# Commands
commands = {
    "end": end,
    "hello": printhello,
    "detect_cutout": detect_cutout,
    "detect_component": detect_component,
    "detect_pcb": detect_pcb,
    "calculate_component_pcb_position": calculate_component_pcb_position,
    "set_component_index": set_component_index,
    "move_arm_over_component": move_arm_over_component,
    "move_head": move_head,
    "reset_arm": reset_arm,
    "show_head_transform": show_head_transform,
}

def main():

    setup(states)

    while True:
        command = input("\nAvailable commands: \n\n" + "\n".join([f"{i}: {s}" for i, s in enumerate(commands.keys())]) + "\n\n")
        print("\n*", command, "*\n")
        try:
            command = int(command)
        
            if command == "end" or command == 0:
                end(states)
            if type(command) is str:
                commands[command](states)
            if type(command) is int:
                list(commands.values())[command](states)
        except:
            print("Invalid Command")
    

if __name__ == "__main__":
    main()
