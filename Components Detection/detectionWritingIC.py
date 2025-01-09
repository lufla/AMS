import cv2
import numpy as np
import easyocr
import re
import time

# Confidence threshold for black square detection
confidenceThreshold = 0.95

def calculate_confidence(contour, mask, frame):
    # Confidence starts at 1.0 (100%)
    confidence = 1

    # Check contour area (larger areas get higher confidence)
    area = cv2.contourArea(contour)
    if area < 400:  # Area too small
        return 0.0
    elif area > 1000:  # Large areas get a boost
        confidence += 0.2

    # Check aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if 0.9 <= aspect_ratio <= 1.1:  # Nearly square
        confidence += 0.2
    else:  # Non-square shapes lose confidence
        confidence -= 0.3

    # Shape similarity (optional but useful)
    square_template = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:  # Only evaluate if 4 vertices
        match_score = cv2.matchShapes(approx, square_template, cv2.CONTOURS_MATCH_I1, 0)
        if match_score < 0.2:  # Lower is better
            confidence += 0.2
        else:
            confidence -= 0.3

    # Color consistency (average pixel value inside contour)
    mask_fill = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(mask_fill, [contour], -1, 255, -1)
    mean_color = cv2.mean(frame, mask=mask_fill)[:3]  # Average BGR color
    if np.mean(mean_color) < 50:  # Dark regions
        confidence += 0.2
    else:
        confidence -= 0.3

    # Clamp confidence to [0, 1]
    return max(0.0, min(1.0, confidence))

def detect_black_squares(frame):
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define black color range
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Optional: Remove noise from the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_ics = []

    for contour in contours:
        # Calculate confidence
        confidence = calculate_confidence(contour, mask, frame)

        if confidence > confidenceThreshold:  # Threshold for accepting detections
            x, y, w, h = cv2.boundingRect(contour)
            detected_ics.append(((x, y, w, h), confidence))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"IC ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, detected_ics

def map_text_to_ic(frame, detected_ics, results):
    for (bbox, text, confidence) in results:
        # Unpack the bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox

        # Convert float coords to int for drawing
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

        # Draw green boundary for detected text
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Find the closest IC to the detected text
        text_center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
        closest_ic = None
        min_distance = float('inf')

        for (ic_bbox, _) in detected_ics:
            x, y, w, h = ic_bbox
            ic_center = (x + w // 2, y + h // 2)
            distance = np.sqrt((text_center[0] - ic_center[0]) ** 2 + (text_center[1] - ic_center[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_ic = ic_bbox

        if closest_ic:
            x, y, w, h = closest_ic
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def main():
    # Initialize the EasyOCR Reader
    reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=True if you have a compatible GPU

    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    # Compile regex for filtering desired text
    pattern = re.compile(r'^IC\d+$')

    while True:
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect black squares in the frame
        frame, detected_ics = detect_black_squares(frame)

        # Convert the frame to grayscale (optional, helps OCR sometimes)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the frame, filtering text during detection
        results = reader.readtext(gray, detail=1)  # Enable detail=1 for bounding boxes

        # Filter and map text to ICs
        filtered_results = [(bbox, text, confidence) for (bbox, text, confidence) in results if pattern.match(text)]
        map_text_to_ic(frame, detected_ics, filtered_results)

        # Calculate and display FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # Show the frame with detection boxes and FPS
        cv2.imshow("Detection and OCR Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
