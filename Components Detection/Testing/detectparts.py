import os
from ultralytics import YOLO
import cv2

def run_webcam_inference(model_path):
    """
    Run inference using a trained YOLO model with webcam input.

    Args:
        model_path (str): Path to the trained YOLO model (e.g., 'best.pt').
    """
    # Load the trained model
    model = YOLO(model_path)

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Run inference on the frame
        results = model.predict(source=frame, show=True, conf=0.25)  # Display results in real-time

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define the path to your trained model
    model_path = "best.pt"  # Replace with the correct path if needed

    # Run webcam inference
    run_webcam_inference(model_path)
