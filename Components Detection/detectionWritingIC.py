import cv2
import numpy as np
import easyocr
import re
import time

# Confidence threshold for black square detection
confidenceThreshold = 0.95

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

    mask_fill = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(mask_fill, [contour], -1, 255, -1)
    mean_color = cv2.mean(frame, mask=mask_fill)[:3]
    if np.mean(mean_color) < 50:
        confidence += 0.2
    else:
        confidence -= 0.3

    return max(0.0, min(1.0, confidence))

def detect_black_squares(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_ics = []

    for contour in contours:
        confidence = calculate_confidence(contour, mask, frame)
        if confidence > confidenceThreshold:
            x, y, w, h = cv2.boundingRect(contour)
            detected_ics.append(((x, y, w, h), confidence))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"IC ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, detected_ics

def map_text_to_ic(frame, detected_ics, results):
    for (bbox, text, confidence) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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

def process_frame_with_rotations(frame, reader, pattern):
    rotations = [0, 90, 180, 270]
    windows = ["Rotated 0", "Rotated 90", "Rotated 180", "Rotated 270"]
    for i, angle in enumerate(rotations):
        if angle == 0:
            rotated_frame = frame.copy()  # No rotation needed
        else:
            rotation_flag = (
                cv2.ROTATE_90_CLOCKWISE if angle == 90 else
                cv2.ROTATE_180 if angle == 180 else
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )
            rotated_frame = cv2.rotate(frame, rotation_flag)

        rotated_frame, detected_ics = detect_black_squares(rotated_frame)
        gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=1)
        filtered_results = [(bbox, text, confidence) for (bbox, text, confidence) in results if pattern.match(text)]
        map_text_to_ic(rotated_frame, detected_ics, filtered_results)

        cv2.imshow(windows[i], rotated_frame)

def main():
    reader = easyocr.Reader(['en'], gpu=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    pattern = re.compile(r'^IC\d+$')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        process_frame_with_rotations(frame, reader, pattern)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
