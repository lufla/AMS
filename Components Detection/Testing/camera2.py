import cv2
import numpy as np

# Define intrinsic camera matrix
K = np.array([
    [629.89396932, 0.0, 303.12864559],
    [0.0, 629.54648765, 262.04433715],
    [0.0, 0.0, 1.0]
])

# Define the perspective matrix (camera calibration matrix)
"""
P = np.array([
    [5.36640295e-01, -2.56710641e-02, 1.71878860e+02],
    [4.72061304e-02, 5.74656957e-01, 2.79284306e+01],
    [-5.09521988e-05, 4.38208610e-05, 1.00000000e+00]
])
"""

P = np.array([[ 2.66901055e-01, -1.28073863e-02,  2.42922577e+02],
 [ 1.21380930e-02,  2.71194859e-01,  2.34352341e+02],
 [-1.15226978e-05, -1.40840655e-05,  1.00000000e+00]])

# Combined intrinsic and perspective correction matrix
M = np.dot(K, P)

# Function to back-project 2D image coordinates to 3D ray
def back_project_to_3d(image_point, M_inv):
    # Transform 2D image coordinates to homogeneous coordinates
    image_homogeneous = np.array([image_point[0], image_point[1], 1])
    ray = np.dot(M_inv, image_homogeneous)
    ray /= ray[2]  # Normalize by Z to ensure proper scaling
    return ray

# Function to calculate depth
def calculate_depth(known_height, image_height, ray_z):
    # known_height: real-world height of the object
    # image_height: height of the object in the image (pixels)
    # ray_z: Z component of the 3D ray
    return known_height / (image_height * ray_z)

# Webcam capture
cap = cv2.VideoCapture(2)  # Adjust the index for your webcam (0, 1, 2, etc.)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Compute the inverse of M
M_inv = np.linalg.inv(M)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Dummy example to detect object and calculate depth (replace with actual object detection)
    h, w, _ = frame.shape
    x_image, y_image = w // 2, h // 2  # Center point of the frame

    # Calculate 3D ray using the perspective matrix
    ray = back_project_to_3d((x_image, y_image), M_inv)

    # Example values for known parameters
    known_height = 0.096  # Real-world height of object in meters
    image_height = 432  # Pixel height of object in the image

    # Calculate depth
    try:
        z_camera = calculate_depth(known_height, image_height, ray[2])
        print(f"Estimated Depth: {z_camera:.10f} meters")
    except ZeroDivisionError:
        print("Error: Division by zero in depth calculation.")
        z_camera = None

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
