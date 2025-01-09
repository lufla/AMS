import cv2
import easyocr
import re
import time

def main():
    # Initialize the EasyOCR Reader.
    reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=True if you have a compatible GPU

    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(2)
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

        # Convert the frame to grayscale (optional, helps OCR sometimes)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the frame, filtering text during detection
        results = reader.readtext(gray, detail=1)  # Enable detail=1 for bounding boxes

        for (bbox, text, confidence) in results:
            if pattern.match(text):
                # Unpack the bounding box
                (top_left, top_right, bottom_right, bottom_left) = bbox

                # Convert float coords to int for drawing
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

                # Draw rectangle around the detected text
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                # Put recognized text (and confidence) on the frame
                cv2.putText(
                    frame,
                    f"{text} ({confidence:.2f})",
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

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
        cv2.imshow("OCR Webcam Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
