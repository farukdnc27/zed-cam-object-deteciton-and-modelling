# ZED Object Detection & Volume Analysis (`haritlama/zed`)

This module integrates the ZED Stereo Camera with **YOLOv8** for real-time object detection and volume estimation. It is designed to identify objects in the scene and calculate their approximate dimensions and volume using depth data.



## 📂 Key Files

- **`zed11.py`**: The main script for object detection and volume calculation. It uses YOLOv8 to detect objects and the ZED depth map to estimate their 3D coordinates and bounding boxes.
- **`yolov8n.pt`**: The pre-trained YOLOv8 Nano model used for detection.
- **`zed2.py` - `zed9.py`**: Various iterations and experimental scripts for testing different detection and depth algorithms.
- **`zed_mesh.obj`**: A sample 3D mesh file, possibly a result of a scanning session.

## ✨ Features

- **Real-time Detection**: Detects common objects (bottles, cups, etc.) using YOLOv8.
- **3D Bounding Boxes**: Draws 3D boxes around detected objects based on their spatial extent.
- **Volume Estimation**: Calculates the volume of the detected objects in cubic units.
- **Distance Measurement**: Displays the distance from the camera to the detected object.

## 🚀 Usage

1.  Ensure you have the required libraries installed:
    ```bash
    pip install ultralytics pyzed-sl opencv-python numpy
    ```

2.  Run the main detection script:
    ```bash
    python zed11.py
    ```

## 📸 Examples



## ⚠️ Notes

- The accuracy of volume estimation depends heavily on the quality of the depth map and the object's material (transparent or reflective surfaces may cause issues).
- Ensure the ZED camera is calibrated correctly.
