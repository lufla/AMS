import cv2
import torch
import numpy as np

# Load MiDaS model
model_type = "DPT_Large"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.to(device)  # Move model to GPU
model.eval()

# Load the model's preprocessing transform
transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

# Initialize webcam
cap = cv2.VideoCapture(2)  # Change the index if you have multiple cameras


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting.")
        break

    # Resize and preprocess the frame for the model
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (384, 384))  # Resize to model input size

    try:
        # Apply transform
        input_batch = transform(input_image).unsqueeze(0)  # Add batch dimension

        # Ensure the tensor is correctly shaped
        if input_batch.ndim != 4 or input_batch.shape[1] != 3:
            print(f"Invalid input tensor shape: {input_batch.shape}. Expected (1, 3, H, W).")
            break

        # Move input to GPU
        input_batch = input_batch.to(device)

        # Perform depth estimation
        with torch.no_grad():
            prediction = model(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

        # Normalize depth map for display
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        depth_map_normalized = np.uint8(depth_map_normalized)

        # Resize depth map back to original frame size
        depth_map_resized = cv2.resize(depth_map_normalized, (frame.shape[1], frame.shape[0]))

        # Combine original frame and depth map
        combined_view = cv2.hconcat([frame, cv2.applyColorMap(depth_map_resized, cv2.COLORMAP_JET)])
        cv2.imshow("Webcam (Left) | Depth Map (Right)", combined_view)

    except Exception as e:
        print(f"Error during processing: {e}")
        break

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
