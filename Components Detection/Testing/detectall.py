import cv2
import time
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
# ----------------------------------------------------------------
# 1. Load the local YOLOv5 models
# ----------------------------------------------------------------

# Get the absolute path to the YOLOv5 directory
# yolov5_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5")
yolov_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultralytics")

# Load the PCB model

# Load the PCB model
model_pcb = YOLO('PCB_componentsV2.pt')  # Path to your weights file

# Load the character model
model_char = YOLO('character-singleV2.pt')  # Path to your weights file
# ----------------------------------------------------------------
# 2. Define a helper function to perform local inference
# ----------------------------------------------------------------
def infer_local(image, model):
    """
    Run YOLOv11 inference on a given frame (NumPy array).
    Returns a list of predictions with x, y, width, height, confidence, class.
    """
    results = model.predict(image)
    predictions = []

    # Iterate over detected boxes
    for box in results[0].boxes:
        # Extract box information
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list
        conf = float(box.conf[0])  # Confidence score
        cls_id = int(box.cls[0])  # Class ID
        label = model.names[cls_id]

        # Convert to desired format
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        predictions.append({
            "x": x_center,
            "y": y_center,
            "width": w,
            "height": h,
            "confidence": conf,
            "class": label
        })

    return predictions

# ----------------------------------------------------------------
# 3. Function to draw boxes on the frame
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
# 4. Main function with video capture and offline inference
# ----------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0
    PROCESS_INTERVAL = 10  # Process every 10th frame

    # For measuring FPS on processed frames
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for faster processing (optional)
        resized_frame = cv2.resize(frame, (320, 240))

        # Only run YOLO inference on every PROCESS_INTERVAL-th frame
        if frame_count % PROCESS_INTERVAL == 0:
            start_time = time.time()

            # We can use a ThreadPool to run both models in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_pcb = executor.submit(infer_local, resized_frame, model_pcb)
                future_characters = executor.submit(infer_local, resized_frame, model_char)

                pcb_predictions = future_pcb.result()
                character_predictions = future_characters.result()

            end_time = time.time()
            fps = 1.0 / (end_time - start_time + 1e-8)

            # Draw the predictions
            draw_predictions(resized_frame, pcb_predictions, color=(0, 255, 0), label_prefix="PCB:")
            draw_predictions(resized_frame, character_predictions, color=(255, 0, 0), label_prefix="Char:")

        # Show FPS on screen (this is the FPS of inference, not your camera feed FPS)
        cv2.putText(resized_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Offline YOLOv5 Detection", resized_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
