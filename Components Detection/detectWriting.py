import cv2
import easyocr
from numpy import character


def main():
    # Initialize the EasyOCR Reader.
    # Specify the list of languages you want to detect.
    # "en" is for English, but you can add others like ["en", "ru"] for English + Russian.
    reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=True if you have a compatible GPU

    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to grayscale (optional, helps OCR sometimes)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the frame
        results = reader.readtext(gray)

        # results is a list of:
        # [ [bbox_coordinates, detected_text, confidence], ... ]

        for (bbox, text, confidence) in results:
            characters = "IC"
            IC = text.startswith(characters)
            # Draw bounding boxes and put recognized text

            if IC:
                # Unpack the bounding box
                (top_left, top_right, bottom_right, bottom_left) = bbox

                # Convert float coords to int for drawing
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

                # Draw rectangle around the detected text
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                # Put recognized text (and maybe confidence) on the frame
                cv2.putText(
                    frame,
                    f"{text} ({confidence:.2f})",
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
            )

            # If you only want to separate letters and digits, you can parse the 'text' string:
            # letters = [c for c in text if c.isalpha()]
            # digits  = [c for c in text if c.isdigit()]
            # Then handle them as needed.

        # Show the frame with detection boxes
        cv2.imshow("OCR Webcam Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
