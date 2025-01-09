import os
from ultralytics import YOLO


def main():
    # Set the correct working directory
    os.chdir('/home/fladlon/GIT/AMS/Components Detection/Testing/Train IC/datasets2')

    # Print debug paths
    print("Current Working Directory:", os.getcwd())
    print("Validation Images Path:", os.path.abspath("valid/images"))
    print("Validation Labels Path:", os.path.abspath("valid/labels"))

    # Dataset YAML path
    data_yaml = "data.yaml"

    # Path to YOLOv8 pre-trained weights
    model_path = "yolov8n.pt"

    # Instantiate YOLO model
    model = YOLO(model_path)

    # Train the model
    model.train(
        data=data_yaml,  # Path to dataset YAML
        epochs=50,  # Number of epochs
        imgsz=640,  # Image size
        batch=8,  # Batch size
        name="pcb-components-yolov8",  # Run name
        project="runs/train",  # Directory for saving results
        val=True  # Enable validation
    )

    # Evaluate the model
    model.val(data=data_yaml)


if __name__ == "__main__":
    main()
