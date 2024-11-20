import os
import re
import cv2
import numpy as np

# Function to extract IC data from the file
def extract_ics(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    ic_data = []
    for line in lines:
        if line.startswith("IC"):
            parts = re.split(r'\s+', line.strip())
            ic_data.append({
                "ref": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "rotation": float(parts[3]),
                "name": parts[4],
                "package": parts[5] if len(parts) > 5 else "Unknown"
            })
    return ic_data

# Relative path to the file
file_name = "PnP_PP3_FPGA_Tester_v3_front.txt"
script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, file_name)

# Extract IC data
ics = extract_ics(file_path)

# Detect ICs in webcam feed
def detect_ic_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_ics = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 0.8 < aspect_ratio < 1.2:  # Heuristic for IC shape
            detected_ics.append((x, y, w, h))
    return detected_ics

def overlay_ic_labels(frame, detected_ics, ic_data):
    for x, y, w, h in detected_ics:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = "Unknown IC"
        # Match with extracted IC data
        for ic in ic_data:
            if abs(ic['x'] - x) < 10 and abs(ic['y'] - y) < 10:  # Adjust tolerance as needed
                label = ic['name']
                break
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Real-time webcam feed processing
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read the frame.")
            break
        detected_ics = detect_ic_contours(frame)
        overlay_ic_labels(frame, detected_ics, ics)
        cv2.imshow("IC Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
