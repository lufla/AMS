import cv2
import time
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="rNI84nUh3sqtBj0Qb1Bs"
)

# Define the model IDs
PCB_MODEL_ID = "pcb-components-r8l8r/13"
CHARACTER_MODEL_ID = "character-single/4"

# Frame processing interval
PROCESS_INTERVAL = 5  # Process every 5th frame


def infer_with_roboflow(image, model_id):
    """
    Send an image to the Roboflow Inference API and return predictions.
    """
    result = CLIENT.infer(image, model_id=model_id)
    return result["predictions"]


def draw_predictions(frame, predictions, color=(0, 255, 0), label_prefix=""):
    """
    Draw bounding boxes and labels on the frame based on predictions.
    """
    for prediction in predictions:
        x = int(prediction["x"] - prediction["width"] / 2)
        y = int(prediction["y"] - prediction["height"] / 2)
        w = int(prediction["width"])
        h = int(prediction["height"])
        conf = prediction["confidence"]
        label = prediction["class"]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Put label text
        label_text = f"{label_prefix}{label} {conf:.2f}"
        cv2.putText(frame, label_text, (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (640, 480))

        # Only process every nth frame
        if frame_count % PROCESS_INTERVAL == 0:
            # Convert frame to RGB for API compatibility
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Save the frame as a temporary file for inference
            temp_image_path = "temp_frame.jpg"
            cv2.imwrite(temp_image_path, rgb_frame)

            start_time = time.time()

            # 1) Infer with the PCB model
            pcb_predictions = infer_with_roboflow(temp_image_path, PCB_MODEL_ID)

            # 2) Infer with the Character model (letters + numbers)
            character_predictions = infer_with_roboflow(temp_image_path, CHARACTER_MODEL_ID)

            end_time = time.time()
            fps = 1 / (end_time - start_time + 1e-8)

            # Draw detection results
            draw_predictions(resized_frame, pcb_predictions, color=(0, 255, 0), label_prefix="PCB:")
            draw_predictions(resized_frame, character_predictions, color=(255, 0, 0), label_prefix="Char:")

        # Show FPS on screen
        cv2.putText(resized_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display the frame
        cv2.imshow("Roboflow Detection", resized_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
