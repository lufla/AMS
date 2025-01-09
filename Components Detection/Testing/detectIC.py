import cv2
import numpy as np
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

    for contour in contours:
        # Calculate confidence
        confidence = calculate_confidence(contour, mask, frame)

        if confidence > confidenceThreshold:  # Threshold for accepting detections
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"IC ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect black squares in the frame
        output_frame = detect_black_squares(frame)

        # Display the result
        cv2.imshow("Black Square Detection", output_frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
