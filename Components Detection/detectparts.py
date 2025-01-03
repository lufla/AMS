import cv2
import numpy as np
import pytesseract


def detect_camera():
    """Automatically detect and connect to the PC camera."""
    cap = cv2.VideoCapture(0)  # Default camera
    if not cap.isOpened():
        print("Camera not detected. Please check the connection.")
        return None
    return cap


def preprocess_image(frame):
    """Preprocess the captured frame for text and grid detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return gray, edges


def detect_grid_and_text(frame, edges):
    """Detect grid and text in the captured frame."""
    # Detect grid lines
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Detect squares (grid cells)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    # Extract text using Tesseract
    text = pytesseract.image_to_string(frame, config='--psm 6')
    print("Detected Text:", text)

    return frame


def main():
    # Detect camera
    cap = detect_camera()
    if not cap:
        return

    print("Press 'q' to quit the camera feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Preprocess the captured frame
        gray, edges = preprocess_image(frame)

        # Detect grid and text
        result_frame = detect_grid_and_text(frame.copy(), edges)

        # Display the results
        cv2.imshow('Detected Grid and Text', result_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
