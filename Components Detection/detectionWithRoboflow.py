import cv2
from ultralytics import YOLO
from roboflow import Roboflow

# Step 1: Initialize Roboflow and YOLO
rf = Roboflow(api_key="<rNI84nUh3sqtBj0Qb1Bs>") # or rf_ebfboIbgNfSAmrfr4RJ7QxhVNEG3
project = rf.workspace().project("printed-circuit-board")
dataset = project.version(4).download("yolov8")
model = YOLO(dataset.model_path)

# Step 2: Initialize Webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or replace with your camera index

if not cap.isOpened():
    print("Error: Unable to access webcam.")
    exit()

def process_frame(frame):
    # Predict on the current frame
    results = model.predict(source=frame, conf=0.5)
    detections = []
    for result in results[0].boxes:
        x1, y1, x2, y2, conf, cls = result.xyxy
        label = result.label
        detections.append({"label": label, "bbox": (int(x1), int(y1), int(x2), int(y2))})

    # Draw detections on the frame
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection["label"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detections

# Step 3: Process Webcam Stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    processed_frame, detections = process_frame(frame)

    # Display the processed frame
    cv2.imshow("Detection", processed_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
