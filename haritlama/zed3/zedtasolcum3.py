import sys
import pyzed.sl as sl
import numpy as np
import open3d as o3d
import time
import traceback
import cv2
import threading


class CornerDetector3D:
    def __init__(self):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.depth_minimum_distance = 300
        init_params.depth_maximum_distance = 5000
        init_params.camera_fps = 30

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("ZED açılırken hata oluştu.")
            exit(1)

        self.runtime_parameters = sl.RuntimeParameters()
        self.point_cloud = sl.Mat()
        self.image = sl.Mat()  # RGB görüntü için
        self.point_list = []
        self.running = True
        self.latest_image = None
        self.image_lock = threading.Lock()

    def get_3d_corners(self):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # 3D nokta bulutunu al
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            
            # RGB görüntüyü al
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            image_np = self.image.get_data()
            
            with self.image_lock:
                self.latest_image = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

            point_cloud_np = self.point_cloud.get_data()
            valid_points = []

            for i in range(point_cloud_np.shape[0]):
                for j in range(point_cloud_np.shape[1]):
                    x, y, z, _ = point_cloud_np[i, j]
                    if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                        valid_points.append([x, y, z])

            return np.array(valid_points)
        return np.array([])

    def show_live_image(self):
        while self.running:
            with self.image_lock:
                if self.latest_image is not None:
                    cv2.imshow("ZED Live View", self.latest_image)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                self.running = False
                break
            time.sleep(0.03)

    def visualize_3d_points(self, points_3d):
        if len(points_3d) == 0:
            print("Görüntüde geçerli 3D nokta bulunamadı.")
            return

        print(f"{len(points_3d)} adet 3D nokta bulundu. Görselleştiriliyor...")

        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(points_3d)

        # Open3D görüntüleme için yeni bir thread başlat
        vis_thread = threading.Thread(
            target=lambda: o3d.visualization.draw_geometries(
                [point_cloud_o3d],
                window_name="ZED 3D Nokta Bulutu",
                width=960,
                height=720
            )
        )
        vis_thread.daemon = True
        vis_thread.start()

    def run(self):
        try:
            # Canlı görüntü için thread başlat
            image_thread = threading.Thread(target=self.show_live_image)
            image_thread.daemon = True
            image_thread.start()

            while self.running:
                points_3d = self.get_3d_corners()
                if len(points_3d) > 0:
                    self.visualize_3d_points(points_3d)
                time.sleep(1)

        except KeyboardInterrupt:
            print("Kapatılıyor...")
        except Exception as e:
            print(f"Hata oluştu: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            cv2.destroyAllWindows()
            self.zed.close()


if __name__ == "__main__":
    detector = CornerDetector3D()
    detector.run()