from roboflow import Roboflow
import os
import cv2
import numpy as np
import tensorflow as tf

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="rNI84nUh3sqtBj0Qb1Bs")
project = rf.workspace("ali-khan-2aqjc").project("hand-safety")
version = project.version(5)
dataset = version.download("yolov5")

# Define dataset paths
train_path = os.path.join(dataset.location, "train")
valid_path = os.path.join(dataset.location, "valid")

# Load images and labels
def load_data(data_path):
    images = []
    labels = []
    for label_file in os.listdir(data_path):
        if label_file.endswith(".jpg") or label_file.endswith(".png"):
            # Load image
            img_path = os.path.join(data_path, label_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # Resize to 224x224
            images.append(img)

            # Load corresponding label (binary classification: safety=1, no_safety=0)
            label_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    label_data = file.readline().strip().split()
                    label = int(label_data[0])  # Assume first value is the class
                    labels.append(label)
            else:
                labels.append(0)  # Default label if no annotation found
    return np.array(images), np.array(labels)

# Load train and validation datasets
x_train, y_train = load_data(train_path)
x_valid, y_valid = load_data(valid_path)

# Normalize image data
x_train = x_train / 255.0
x_valid = x_valid / 255.0

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
])

# Compile the model
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))

# Save the model
model.save("hand_safety_model.h5")
print("Model saved as 'hand_safety_model.h5'")
