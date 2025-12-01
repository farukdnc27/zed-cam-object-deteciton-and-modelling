import sys
import pyzed.sl as sl
import numpy as np
import open3d as o3d
import time
import traceback

class StoneDimensionEstimator:
    def __init__(self):
        # --- ZED Kamera Kurulumu (Değişiklik yok) ---
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

    def get_point_cloud_from_zed(self):
        # --- Bu fonksiyon aynı kaldı ---
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            point_cloud_np = self.point_cloud.get_data()
            points = point_cloud_np[:, :, :3]
            mask = np.isfinite(points).all(axis=2)
            valid_points = points[mask]
            return valid_points
        return None

    def process_point_cloud(self, points_np, voxel_size=5.0):
        # --- Bu fonksiyon aynı kaldı ---
        if points_np is None or len(points_np) == 0:
            return None, None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        plane_model, inliers = pcd_clean.segment_plane(distance_threshold=15.0, ransac_n=3, num_iterations=1000)
        objects_pcd = pcd_clean.select_by_index(inliers, invert=True)
        if len(objects_pcd.points) == 0:
            return pcd_clean, None
        labels = np.array(objects_pcd.cluster_dbscan(eps=25.0, min_points=15, print_progress=False))
        if labels.max() == -1:
            return pcd_clean, None
        largest_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))
        stone_indices = np.where(labels == largest_cluster_label)[0]
        stone_pcd = objects_pcd.select_by_index(stone_indices)
        return pcd_clean, stone_pcd

    def calculate_measurements(self, stone_pcd):
        # --- Bu fonksiyon aynı kaldı ---
        if stone_pcd is None or len(stone_pcd.points) < 10:
            return None
        obb = stone_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)
        dimensions = obb.extent
        try:
            hull, _ = stone_pcd.compute_convex_hull()
            volume = hull.get_volume()
            hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            hull_ls.paint_uniform_color((0, 1, 0))
        except Exception:
            volume = 0
            hull_ls = None
        return {
            "dimensions_mm": dimensions,
            "volume_mm3": volume,
            "oriented_bounding_box": obb,
            "convex_hull_lineset": hull_ls
        }

    # YENİ FONKSİYON: Taştan katı model (mesh) oluşturur
    def create_stone_mesh(self, stone_pcd, alpha=20.0):
        """
        Nokta bulutundan Alpha Shape algoritması ile bir 3D yüzey (mesh) oluşturur.
        """
        if stone_pcd is None or len(stone_pcd.points) < 20:
            return None
        
        # Alpha-shape algoritmasını kullanarak mesh oluştur
        # 'alpha' değeri, detayı belirler. Düşük alpha daha detaylı, yüksek alpha daha genel bir şekil oluşturur.
        # Bu değeri taşınızın nokta yoğunluğuna göre ayarlamanız gerekebilir.
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(stone_pcd, alpha)
            # Mesh'in daha pürüzsüz ve katı görünmesi için yüzey normallerini hesapla
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.5, 0.5, 1.0]) # Açık mavi bir renk verelim
            return mesh
        except Exception as e:
            print(f"Mesh oluşturulamadı: {e}")
            return None

    def run(self):
        """Ana döngü - Görselleştirme iyileştirmeleri ile güncellendi"""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("Taş Boyut ve Hacim Tespiti", width=1280, height=720)
        
        # Arka planı gizlemek/göstermek için bir durum değişkeni
        state = {"show_background": True}

        # 'B' tuşuna basıldığında çağrılacak fonksiyon
        def toggle_background(vis):
            state["show_background"] = not state["show_background"]
            print(f"Arka plan gösterimi: {'AÇIK' if state['show_background'] else 'KAPALI'}")
            # Değişikliği anında uygulamak için bir güncelleme tetikle
            # Bu kısım doğrudan bir 'update' gerektirmez, döngü bir sonraki adımda halleder.
            return False

        # 'B' tuşunu (B for Background) callback fonksiyonuna ata
        vis.register_key_callback(ord("B"), toggle_background)

        geometries_added = False
        
        try:
            while True:
                points_np = self.get_point_cloud_from_zed()
                if points_np is None: continue

                original_pcd, stone_pcd = self.process_point_cloud(points_np)
                if original_pcd is None: continue

                measurements = self.calculate_measurements(stone_pcd)
                
                # YENİ ADIM: Taşın katı modelini oluştur
                stone_mesh = self.create_stone_mesh(stone_pcd, alpha=25.0)

                if measurements:
                    dims = measurements["dimensions_mm"]
                    vol = measurements["volume_mm3"]
                    print(f"--- TAŞ ÖLÇÜMLERİ ---")
                    print(f"Boyutlar (mm): U={dims[0]:.1f}, G={dims[1]:.1f}, Y={dims[2]:.1f}")
                    print(f"Hacim (Convex Hull): {vol / 1e6:.2f} cm³")
                    print("-" * 25)

                # -- GÜNCELLENMİŞ GÖRSELLEŞTİRME --
                vis.clear_geometries()
                
                # Arka planı sadece 'show_background' True ise ekle
                if state["show_background"]:
                    vis.add_geometry(original_pcd, reset_bounding_box=not geometries_added)
                
                if measurements:
                    # Artık mavi nokta bulutu yerine katı modeli (mesh) ekliyoruz
                    if stone_mesh:
                        vis.add_geometry(stone_mesh, reset_bounding_box=not geometries_added)
                    # Eğer mesh oluşturulamazsa, eski usul nokta bulutunu göster
                    elif stone_pcd:
                        stone_pcd.paint_uniform_color([0, 0, 1])
                        vis.add_geometry(stone_pcd, reset_bounding_box=not geometries_added)

                    vis.add_geometry(measurements["oriented_bounding_box"], reset_bounding_box=not geometries_added)
                    if measurements["convex_hull_lineset"]:
                        vis.add_geometry(measurements["convex_hull_lineset"], reset_bounding_box=not geometries_added)
                
                geometries_added = True
                vis.poll_events()
                vis.update_renderer()

        except KeyboardInterrupt:
            print("Kapatılıyor...")
        except Exception as e:
            print(f"Bir hata oluştu: {e}")
            traceback.print_exc()
        finally:
            vis.destroy_window()
            self.zed.close()

if __name__ == "__main__":
    detector = StoneDimensionEstimator()
    detector.run()