import sys
import pyzed.sl as sl
import numpy as np
import open3d as o3d
import time
import traceback
import cv2 # OpenCV'yi import ediyoruz

# Önceki kodumuzdan StoneDimensionEstimator sınıfını buraya kopyalayın.
# Hiçbir değişiklik yapmanıza gerek yok.
class StoneDimensionEstimator:
    def __init__(self):
        # ... (Önceki kodla aynı)
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.depth_minimum_distance = 300
        init_params.depth_maximum_distance = 5000
        init_params.camera_fps = 30

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("ZED kamerasını açarken hata oluştu.")
            sys.exit(1)

        self.runtime_parameters = sl.RuntimeParameters()
        self.point_cloud = sl.Mat()
        # YENİ: Renkli görüntü için Mat nesnesi
        self.image = sl.Mat()


    def get_point_cloud_and_image(self):
        """ZED'den hem nokta bulutunu hem de renkli görüntüyü alır."""
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Nokta bulutunu al
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            point_cloud_np = self.point_cloud.get_data()
            points = point_cloud_np[:, :, :3]
            mask = np.isfinite(points).all(axis=2)
            valid_points = points[mask]
            
            # Renkli görüntüyü al
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            image_np = self.image.get_data()
            # RGBA'dan BGR'a çevir (OpenCV formatı)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

            return valid_points, image_bgr
        return None, None

    def process_point_cloud(self, points_np, voxel_size=5.0):
        # ... (Önceki kodla aynı)
        if points_np is None or len(points_np) == 0: return None, None
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        plane_model, inliers = pcd_clean.segment_plane(distance_threshold=15.0, ransac_n=3, num_iterations=1000)
        objects_pcd = pcd_clean.select_by_index(inliers, invert=True)
        if len(objects_pcd.points) == 0: return pcd_clean, None
        labels = np.array(objects_pcd.cluster_dbscan(eps=25.0, min_points=25, print_progress=False))
        if labels.max() == -1: return pcd_clean, None
        counts = np.bincount(labels[labels >= 0])
        if len(counts) == 0: return pcd_clean, None
        largest_cluster_label = np.argmax(counts)
        stone_indices = np.where(labels == largest_cluster_label)[0]
        stone_pcd = objects_pcd.select_by_index(stone_indices)
        return pcd_clean, stone_pcd

    def calculate_measurements(self, stone_pcd):
        # ... (Önceki kodla aynı, sadece bonus ağırlık ekledim)
        if stone_pcd is None or len(stone_pcd.points) < 10: return None
        obb = stone_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)
        dimensions = obb.extent
        try:
            hull, _ = stone_pcd.compute_convex_hull()
            volume = hull.get_volume() # mm^3
        except Exception:
            volume = 0
        
        # BONUS: Ağırlık tahmini (Bazalt yoğunluğu ~2.9 g/cm³)
        stone_density_g_cm3 = 2.9
        volume_cm3 = volume / 1000
        weight_kg = (volume_cm3 * stone_density_g_cm3) / 1000

        return {
            "dimensions_mm": dimensions,
            "volume_cm3": volume_cm3,
            "weight_kg": weight_kg,
            "oriented_bounding_box": obb,
        }

    def create_stone_mesh(self, stone_pcd, alpha=25.0):
        # ... (Önceki kodla aynı)
        if stone_pcd is None or len(stone_pcd.points) < 20: return None
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(stone_pcd, alpha)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.7, 0.7, 0.7]) # Metalik gri bir renk
            return mesh
        except Exception: return None

    # TAMAMEN YENİ, MÜŞTERİYE ÖZEL SUNUM FONKSİYONU
    def run_customer_demo(self):
        """Müşteriye özel, temiz ve profesyonel bir sunum arayüzü çalıştırır."""
        
        # 3D GÖRSELLEŞTİRİCİ (SAĞ-ÜST)
        vis = o3d.visualization.Visualizer()
        vis.create_window("3D Analiz", width=640, height=480, left=650, top=50)
        vis.get_render_option().background_color = np.asarray([0.15, 0.15, 0.18]) # Koyu arkaplan
        vis.get_render_option().mesh_show_back_face = True

        # 3D modelin dönüşü için
        view_control = vis.get_view_control()
        
        try:
            while True:
                # 1. ZED'den hem görüntü hem nokta bulutu al
                points_np, image_bgr = self.get_point_cloud_and_image()
                if points_np is None:
                    continue

                # 2. Nokta bulutunu işle
                _, stone_pcd = self.process_point_cloud(points_np)
                
                # 3. Ölçümleri ve 3D Modeli oluştur
                measurements = self.calculate_measurements(stone_pcd)
                stone_mesh = self.create_stone_mesh(stone_pcd)

                # --- OpenCV Arayüzü (SOL TARAF) ---
                # Çıktıları yazmak için görüntü üzerine siyah bir panel ekle
                panel_height = 150
                cv2.rectangle(image_bgr, (0, image_bgr.shape[0] - panel_height), (image_bgr.shape[1], image_bgr.shape[0]), (0,0,0), -1)

                cv2.putText(image_bgr, "CANLI GORUNTU", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if measurements:
                    dims = measurements["dimensions_mm"]
                    vol = measurements["volume_cm3"]
                    wgt = measurements["weight_kg"]
                    
                    # Sonuçları cm cinsinden yazdır
                    dim_text = f"BOYUTLAR (cm): {dims[0]/10:.1f} x {dims[1]/10:.1f} x {dims[2]/10:.1f}"
                    vol_text = f"HACIM (cm3): {vol:.1f}"
                    wgt_text = f"TAHMINI AGIRLIK (kg): {wgt:.2f}"
                    
                    cv2.putText(image_bgr, dim_text, (10, image_bgr.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(image_bgr, vol_text, (10, image_bgr.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(image_bgr, wgt_text, (10, image_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("Kamera Goruntusu ve Sonuclar", image_bgr)
                
                # --- Open3D Arayüzü (SAĞ TARAF) ---
                vis.clear_geometries()
                if stone_mesh:
                    vis.add_geometry(stone_mesh)
                    # Modeli kendi etrafında yavaşça döndür
                    view_control.rotate(5.0, 0.0) 
                
                vis.poll_events()
                vis.update_renderer()

                # Çıkış için 'q' tuşunu bekle
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            vis.destroy_window()
            cv2.destroyAllWindows()
            self.zed.close()

if __name__ == "__main__":
    detector = StoneDimensionEstimator()
    # Eski run fonksiyonu yerine yeni müşteri demosunu çalıştır
    detector.run_customer_demo()