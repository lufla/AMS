import cv2
import numpy as np
import pandas as pd
import os
from gerber import load_layer


def load_component_data(front_file, back_file, other_files):
    # Load component data from front, back, and other files
    front_data = pd.read_csv(front_file, delimiter='\t', header=None,
                             names=['ID', 'X', 'Y', 'Rotation', 'Value', 'Package'])
    back_data = pd.read_csv(back_file, delimiter='\t', header=None,
                            names=['ID', 'X', 'Y', 'Rotation', 'Value', 'Package'])
    all_data = [front_data, back_data]

    # Load other component or pad data from additional files
    for file in other_files:
        if file.endswith('.txt'):
            data = pd.read_csv(file, delimiter='\t', header=None,
                               names=['ID', 'X', 'Y', 'Rotation', 'Value', 'Package'])
            all_data.append(data)

    # Combine all data into one DataFrame and drop rows with missing values
    combined_data = pd.concat(all_data, ignore_index=True).dropna(subset=['X', 'Y'])
    return combined_data


def parse_gerber_files(gerber_files):
    pads = []
    for file in gerber_files:
        if file.endswith('.gbr') or file.endswith('.xln'):
            layer = load_layer(file)
            for primitive in layer.primitives:
                if hasattr(primitive, 'position'):
                    pads.append((primitive.position[0], primitive.position[1]))
    return pads


def detect_shapes(frame, overlay, components, pads):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours from edges detected
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to reduce the number of points
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

        # Filter out small or very large contours to reduce clutter
        area = cv2.contourArea(contour)
        if area < 100 or area > 5000:
            continue

        # Draw bounding shapes for squares and circles
        if len(approx) == 4:  # Square or Rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:  # Ensure it is a square
                cv2.drawContours(overlay, [approx], 0, (0, 0, 255), 3)  # Red color, thicker line
                cv2.putText(overlay, 'Square', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif len(approx) > 7:  # Circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if 10 < radius < 50:
                cv2.circle(overlay, (int(x), int(y)), int(radius), (255, 0, 255), 3)  # Magenta color, thicker line
                cv2.putText(overlay, 'Circle', (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Overlay component data on the frame
    for _, row in components.iterrows():
        comp_x, comp_y = int(row['X']), int(row['Y'])
        # Adjust component coordinates to align properly with the frame
        frame_height, frame_width = frame.shape[:2]
        adjusted_x = int(comp_x * frame_width / 100)  # Assuming coordinates are in percentage
        adjusted_y = int(comp_y * frame_height / 100)
        # Mirror the x-coordinate for the overlay
        mirrored_x = frame_width - adjusted_x
        cv2.putText(overlay, row['ID'], (mirrored_x, adjusted_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green color, thicker text
        cv2.circle(overlay, (mirrored_x, adjusted_y), 5, (0, 255, 0), -1)  # Green color, larger circle

    # Overlay pad data from Gerber files
    for pad_x, pad_y in pads:
        # Adjust pad coordinates to align properly with the frame
        adjusted_x = int(pad_x * frame_width / 100)
        adjusted_y = int(pad_y * frame_height / 100)
        # Mirror the x-coordinate for the overlay
        mirrored_x = frame_width - adjusted_x
        cv2.circle(overlay, (mirrored_x, adjusted_y), 5, (255, 255, 0), -1)  # Yellow color, larger circle


def detect_pcb(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny Edge Detection to find edges in the frame
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours from edges detected
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find the largest rectangular contour
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Look for quadrilateral shapes (rectangles)
                largest_contour = approx
                max_area = area

    # If a PCB-like contour is found, return the bounding box with improved certainty
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Expand the bounding box slightly to ensure full PCB is included
        padding = 10
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, frame.shape[1] - x)
        h = min(h + 2 * padding, frame.shape[0] - y)
        return x, y, w, h
    else:
        return None


def save_overlay_image(frame, overlay, output_file):
    # Combine the original frame with the overlay
    combined = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    # Save the resulting image
    cv2.imwrite(output_file, combined)


# Main function to run webcam and detect shapes
def main():
    # Load component data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    front_file = os.path.join(script_dir, 'PnP_PP3_FPGA_Tester_v3_front.txt')
    back_file = os.path.join(script_dir, 'PnP_PP3_FPGA_Tester_v3_back.txt')
    other_files = [
        os.path.join(script_dir, 'PP3_FPGA_Tester_v3.txt'),
    ]
    gerber_files = [
        os.path.join(script_dir, 'copper_top.gbr'),
        os.path.join(script_dir, 'drill_1_16.xln'),
        os.path.join(script_dir, 'profile.gbr'),
        os.path.join(script_dir, 'silkscreen_top.gbr'),
        os.path.join(script_dir, 'soldermask_top.gbr'),
        os.path.join(script_dir, 'solderpaste_top.gbr')
    ]

    # Load component data (front, back, and additional data)
    combined_data = load_component_data(front_file, back_file, other_files)

    # Parse Gerber files to extract pad locations
    pads = parse_gerber_files(gerber_files)

    # Start the video capture from the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the width of the window
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the height of the window
    overlay = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Detect PCB in the frame
        pcb_bbox = detect_pcb(frame)

        if pcb_bbox is not None:
            x, y, w, h = pcb_bbox
            # Crop the frame to the detected PCB area
            pcb_frame = frame[y:y+h, x:x+w]

            # Initialize the overlay image with the same size as the PCB area
            overlay = np.zeros_like(pcb_frame)

            # Detect and visualize shapes in the cropped PCB area
            detect_shapes(pcb_frame, overlay, combined_data, pads)

            # Combine the cropped PCB frame with the overlay
            combined = cv2.addWeighted(pcb_frame, 0.8, overlay, 0.2, 0)

            # Place the combined overlay back on the original frame
            frame[y:y+h, x:x+w] = combined

        # Display the resulting frame
        cv2.imshow('Shape Detection and Component Matching', frame)

        # Press 's' to save the current frame with overlay
        if cv2.waitKey(1) & 0xFF == ord('s'):
            output_file = os.path.join(script_dir, 'matched_components.png')
            save_overlay_image(frame, overlay, output_file)
            print(f"Saved matched components to {output_file}")

        # Press 'q' to quit the webcam window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when everything is done
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
