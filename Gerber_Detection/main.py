import os
import cv2
import numpy as np
import gerber
from gerber.render.cairo_backend import GerberCairoContext
import platform


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

    # Determine the bounding box for the PCB based on all elements
    for outline in outlines:
        x, y = outline.position
        width, height = outline.width, outline.height
        min_x, min_y = min(min_x, x - width / 2), min(min_y, y - height / 2)
        max_x, max_y = max(max_x, x + width / 2), max(max_y, y + height / 2)

    padding = 5  # Small padding around the edges of the PCB
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    width = int((max_x - min_x) * 90)
    height = int((max_y - min_y) * 90)
    img = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA for transparency

    # Draw outlines and components with vibrant colors
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
        cv2.circle(img, center, radius, (255, 0, 0, 200), -1)  # Vibrant blue

    for hole in holes:
        center = (int((hole.position[0] - min_x) * 90), int((hole.position[1] - min_y) * 90))
        radius = int(hole.diameter / 4 * 90)
        cv2.circle(img, center, radius, (0, 255, 255, 200), -1)  # Vibrant yellow-green

    for line in pcb_profile:
        start = (int((line.start[0] - min_x) * 90), int((line.start[1] - min_y) * 90))
        end = (int((line.end[0] - min_x) * 90), int((line.end[1] - min_y) * 90))
        cv2.line(img, start, end, (255, 255, 255, 200), 3)  # White lines for profile

    cv2.imwrite(output_file, img)


# Function to overlay the image on the PCB detected in the foreground
def show_overlay_with_webcam(overlay_image_path):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system() == 'Windows' else 0)
    if not cap.isOpened():
        raise IOError("Could not open webcam.")

    # Load and mirror the overlay image
    overlay = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
    overlay = cv2.flip(overlay, 1)  # Flip horizontally

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to HSV for color-based PCB detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])  # Adjust these values to detect the PCB color
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect largest contour assuming it is the PCB
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            resized_overlay = cv2.resize(overlay, (w, h))

            # Overlay the transparent image on the PCB area
            overlay_alpha = resized_overlay[:, :, 3] / 255.0
            for c in range(0, 3):
                frame[y:y+h, x:x+w, c] = (overlay_alpha * resized_overlay[:, :, c] +
                                          (1 - overlay_alpha) * frame[y:y+h, x:x+w, c])

        cv2.imshow('Webcam with Overlay', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


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
