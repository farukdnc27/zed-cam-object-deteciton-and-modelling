# -*- coding: utf-8 -*-
"""
ZED 2i Perfect 3D Volume Measurement
------------------------------------
This script is designed to accurately measure the volume of an object using ZED 2i stereo camera.
"""

import pyzed.sl as sl
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyransac3d as pyrsc


# --- 1. CONFIGURATION (All settings here) ---
class Config:
    """Manages algorithm and camera parameters centrally."""
    # ZED Camera Settings
    RESOLUTION = sl.RESOLUTION.HD720
    FPS = 30
    DEPTH_MODE = sl.DEPTH_MODE.ULTRA
    COORDINATE_UNITS = sl.UNIT.METER
    MAX_DEPTH = 5.0

    # ZED Runtime Settings
    CONFIDENCE_THRESHOLD = 95
    TEXTURE_CONFIDENCE_THRESHOLD = 95

    # Point Cloud Processing Settings
    DOWNSAMPLING_FACTOR = 4

    # Floor Removal Settings
    FLOOR_REMOVAL_ENABLED = True
    FLOOR_PLANE_THRESHOLD = 0.02

    # Object Segmentation Settings (DBSCAN)
    DBSCAN_EPS = 0.04
    DBSCAN_MIN_SAMPLES = 50

    # Volume Calculation Settings
    VOXEL_SIZE = 0.005

    # Visualization
    VISUALIZE_3D_POINTS = True


# --- 2. CAMERA MANAGER ---
class ZEDManager:
    """Manages all operations with ZED camera."""

    def __init__(self, cfg: Config):
        self.zed = sl.Camera()
        self.config = cfg
        self.init_params = self._setup_init_parameters()
        self.runtime_params = self._setup_runtime_parameters()

    def _setup_init_parameters(self):
        params = sl.InitParameters()
        params.camera_resolution = self.config.RESOLUTION
        params.camera_fps = self.config.FPS
        params.depth_mode = self.config.DEPTH_MODE
        params.coordinate_units = self.config.COORDINATE_UNITS
        params.depth_maximum_distance = self.config.MAX_DEPTH
        return params

    def _setup_runtime_parameters(self):
        params = sl.RuntimeParameters()
        params.confidence_threshold = self.config.CONFIDENCE_THRESHOLD
        params.texture_confidence_threshold = self.config.TEXTURE_CONFIDENCE_THRESHOLD
        return params

    def open(self):
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"ERROR: Camera could not be opened. Error Code: {err}")
            return False
        print("✅ ZED 2i camera successfully initialized.")
        return True

    def grab_data(self):
        image = sl.Mat()
        point_cloud = sl.Mat()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            return image.get_data(), point_cloud
        return None, None

    def close(self):
        self.zed.close()
        print("Camera closed.")


# --- 3. POINT CLOUD PROCESSOR ---
class PointCloudProcessor:
    """Processes point cloud to calculate volume."""

    def __init__(self, cfg: Config):
        self.config = cfg

    def process(self, point_cloud_mat, roi):
        """Manages the main processing flow."""
        print("\n--- Starting New Volume Calculation ---")

        points = self._extract_points_from_roi(point_cloud_mat, roi)
        if points is None: return None

        if self.config.FLOOR_REMOVAL_ENABLED:
            points = self._remove_floor_plane(points)
            if points is None: return None

        object_points = self._segment_object(points)
        if object_points is None: return None

        volume_data = self._calculate_volume(object_points)

        if self.config.VISUALIZE_3D_POINTS:
            self._visualize_points(points, object_points)

        return volume_data

    def _extract_points_from_roi(self, point_cloud_mat, roi):
        """Gets valid 3D points within the selected rectangle (ROI)."""
        x1, y1, x2, y2 = roi
        all_points = point_cloud_mat.get_data()

        roi_points_with_color = all_points[y1:y2, x1:x2]

        points_3d = roi_points_with_color[:, :, :3]
        mask = np.isfinite(points_3d).all(axis=2)
        valid_points = points_3d[mask]

        if valid_points.shape[0] < 100:
            print(f"❌ ERROR: Not enough points in ROI ({valid_points.shape[0]} points).")
            return None

        if self.config.DOWNSAMPLING_FACTOR > 1:
            valid_points = valid_points[::self.config.DOWNSAMPLING_FACTOR, :]

        print(f"📊 Extracted {valid_points.shape[0]} points from ROI.")
        return valid_points

    def _remove_floor_plane(self, points):
        """Finds floor plane with RANSAC and removes points belonging to this plane."""
        try:
            plane = pyrsc.Plane()
            best_eq, best_inliers = plane.fit(points, thresh=self.config.FLOOR_PLANE_THRESHOLD, maxIteration=100)

            object_points = np.delete(points, best_inliers, axis=0)

            print(
                f"✅ Floor removed: {len(best_inliers)} points identified as floor. {object_points.shape[0]} points remaining.")
            return object_points
        except Exception as e:
            print(f"⚠️ Warning: Floor removal failed. Continuing... ({e})")
            return points

    def _segment_object(self, points):
        """Finds the largest point cluster (object) using DBSCAN."""
        if points.shape[0] < self.config.DBSCAN_MIN_SAMPLES:
            print("❌ ERROR: Not enough points for clustering.")
            return None

        db = DBSCAN(eps=self.config.DBSCAN_EPS, min_samples=self.config.DBSCAN_MIN_SAMPLES).fit(points)
        labels = db.labels_

        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) == 0:
            print("❌ ERROR: No cluster that can be identified as an object was found.")
            return None

        largest_cluster_label = unique_labels[np.argmax(counts)]
        object_points = points[labels == largest_cluster_label]

        print(f"🎯 Object successfully isolated with {object_points.shape[0]} points.")
        return object_points

    def _calculate_volume(self, points):
        """Calculates volume using voxel grid method."""
        min_coords = np.min(points, axis=0)

        voxel_indices = ((points - min_coords) / self.config.VOXEL_SIZE).astype(int)
        unique_voxels = np.unique(voxel_indices, axis=0)

        volume = len(unique_voxels) * (self.config.VOXEL_SIZE ** 3)
        dimensions = np.max(points, axis=0) - np.min(points, axis=0)

        return {"volume_m3": volume, "dimensions_m": dimensions, "point_count": len(points)}

    def _visualize_points(self, all_points, object_points):
        """Visualizes calculation steps in 3D."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='lightgray', s=1, alpha=0.1,
                   label='Background')
        ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c=object_points[:, 2], cmap='viridis',
                   s=3, label='Object')

        ax.set_title('Object Segmentation Result')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.axis('equal')
        plt.show()


# --- 4. USER INTERFACE MANAGER ---
class UIManager:
    """Manages OpenCV window, mouse events and drawing operations."""

    def __init__(self, window_name="ZED Volume Measurement"):
        self.window_name = window_name
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.roi_complete = None
        self.request_process = False

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi_start = (x, y)
            self.roi_end = (x, y)
            self.roi_complete = None
            self.request_process = False

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            self.roi_complete = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.request_process = True

    def update_display(self, frame, results):
        """Adds information and drawings on the image."""
        try:
            if frame is None or frame.size == 0:
                return

            # Ensure frame is in BGR format and writable
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Make a copy to ensure we're not modifying a read-only frame
            display_frame = frame.copy()

            if self.drawing:
                cv2.rectangle(display_frame, self.roi_start, self.roi_end, (0, 255, 255), 2)
            elif self.roi_complete:
                x1, y1, x2, y2 = self.roi_complete
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            try:
                cv2.putText(display_frame, "Draw a rectangle around the object.", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "'c': Clear | 'q': Quit", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            except Exception as e:
                print(f"⚠️ Warning: Could not add text to frame: {e}")

            if results:
                vol_liters = results["volume_m3"] * 1000
                dims = results["dimensions_m"]
                text_vol = f"Volume: {vol_liters:.2f} liters"
                text_dim = f"Dimensions: {dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f} m"
                try:
                    cv2.putText(display_frame, text_vol, (10, display_frame.shape[0] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display_frame, text_dim, (10, display_frame.shape[0] - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    print(f"⚠️ Warning: Could not add results text to frame: {e}")

            cv2.imshow(self.window_name, display_frame)
        except Exception as e:
            print(f"⚠️ Error in display update: {e}")

    def clear(self):
        self.roi_start = self.roi_end = self.roi_complete = None
        self.drawing = self.request_process = False
        print("Screen cleared.")


# --- 5. MAIN APPLICATION ---
def main():
    print(__doc__)
    print("Required Libraries: opencv-python, numpy, scikit-learn, matplotlib, pyransac3d")

    config = Config()
    zed = ZEDManager(config)
    processor = PointCloudProcessor(config)
    ui = UIManager()

    if not zed.open():
        return

    results = None

    try:
        while True:
            frame, point_cloud = zed.grab_data()
            if frame is None or point_cloud is None or frame.size == 0:
                print("❌ Could not get camera data. Waiting...")
                cv2.waitKey(100)  # wait a bit to slow down the loop
                continue

            if ui.request_process:
                results = processor.process(point_cloud, ui.roi_complete)
                ui.request_process = False

            ui.update_display(frame, results)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                ui.clear()
                results = None
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"⚠️ Fatal error: {e}")
    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()