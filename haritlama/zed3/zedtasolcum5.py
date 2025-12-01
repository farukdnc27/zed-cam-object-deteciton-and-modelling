import sys
import os
import datetime
import traceback

import numpy as np
import cv2
import open3d as o3d
# YENİ: Open3D'nin modern GUI ve Rendering modüllerini import ediyoruz
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pyzed.sl as sl


class StoneDimensionEstimator:
    # __init__, get_point_cloud_and_image, process_point_cloud,
    # calculate_measurements, create_stone_mesh, project_3d_to_2d
    # fonksiyonları BİREBİR AYNI KALIYOR.
    # Değişiklik sadece run_customer_demo fonksiyonunda olacak.
    # Okunabilirlik için o fonksiyonları buraya tekrar ekliyorum.

    def __init__(self):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.depth_minimum_distance = 400
        init_params.depth_maximum_distance = 5000
        init_params.camera_fps = 30
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("ZED kamerası açılamadı!")
            sys.exit(1)

        self.runtime_parameters = sl.RuntimeParameters()
        self.point_cloud = sl.Mat()
        self.image = sl.Mat()

        cam_info = self.zed.get_camera_information()
        self.camera_intrinsics = cam_info.camera_configuration.calibration_parameters.left_cam
        print("Kamera intrinsic parametreleri başarıyla alındı.")

    def get_point_cloud_and_image(self):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            pc = self.point_cloud.get_data()
            points = pc[:, :, :3].astype(np.float32)
            mask = np.isfinite(points).all(axis=2)
            valid_points = points[mask]

            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            img = self.image.get_data()
            image_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return valid_points, image_bgr
        return None, None

    def process_point_cloud(self, points_np, voxel_size=5.0):
        if points_np is None or len(points_np) == 0: return None, None
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        if len(pcd_down.points) == 0: return pcd_down, None
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        try:
            plane_model, inliers = pcd_clean.segment_plane(distance_threshold=15.0, ransac_n=3, num_iterations=1000)
            objects_pcd = pcd_clean.select_by_index(inliers, invert=True)
        except Exception: objects_pcd = pcd_clean
        if len(objects_pcd.points) == 0: return pcd_clean, None
        labels = np.array(objects_pcd.cluster_dbscan(eps=25.0, min_points=25, print_progress=False))
        if labels.size == 0 or labels.max() == -1: return pcd_clean, None
        counts = np.bincount(labels[labels >= 0])
        if counts.size == 0: return pcd_clean, None
        largest_cluster_label = np.argmax(counts)
        stone_indices = np.where(labels == largest_cluster_label)[0]
        stone_pcd = objects_pcd.select_by_index(stone_indices)
        return pcd_clean, stone_pcd

    def calculate_measurements(self, stone_pcd):
        if stone_pcd is None or len(stone_pcd.points) < 10: return None
        obb = stone_pcd.get_oriented_bounding_box()
        dimensions = np.asarray(obb.extent)
        volume_mm3 = 0.0
        try:
            hull, _ = stone_pcd.compute_convex_hull()
            if hull.is_watertight(): volume_mm3 = hull.get_volume()
        except Exception: pass
        volume_cm3 = float(volume_mm3) / 1000.0
        stone_density_g_cm3 = 2.7
        weight_kg = (volume_cm3 * stone_density_g_cm3) / 1000.0
        return {
            "dimensions_mm": dimensions,
            "volume_cm3": volume_cm3,
            "weight_kg": weight_kg,
            "oriented_bounding_box": obb,
        }

    def create_stone_mesh(self, stone_pcd, alpha=25.0):
        if stone_pcd is None or len(stone_pcd.points) < 20: return None
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(stone_pcd, alpha)
        except Exception:
            try: hull, _ = stone_pcd.compute_convex_hull(); mesh = hull
            except Exception: return None
        mesh.compute_vertex_normals()
        return mesh
    
    # TAMAMEN YENİ VE GÜNCELLENMİŞ SUNUM FONKSİYONU
    def run_customer_demo(self):
        # GUI Uygulamasını başlat
        gui.Application.instance.initialize()

        # Pencere ve Sahne Widget'ı oluştur
        w = gui.Application.instance.create_window("Taş Analiz Sistemi", 1280, 800)
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(w.renderer)
        w.add_child(scene_widget)

        # Durum değişkenleri (class seviyesinde veya bir sözlük içinde)
        self.mode = "realtime"
        self.last_data = {
            "mesh": None,
            "obb": None,
            "measurements": None,
            "image": np.zeros((720, 1280, 3), dtype=np.uint8)
        }

        # Sahne materyalleri
        mat_stone = rendering.MaterialRecord()
        mat_stone.base_color = [0.7, 0.7, 0.7, 1.0]
        mat_stone.shader = "defaultLit"

        mat_box = rendering.MaterialRecord()
        mat_box.base_color = [1.0, 0.0, 0.0, 1.0]
        mat_box.shader = "unlitLine"
        mat_box.line_width = 3.0

        def update_scene():
            """Sahneyi temizler ve en son verilerle yeniden doldurur."""
            scene = scene_widget.scene
            scene.clear_geometry()
            
            # 3D etiketleri temizlemek için eski usul (henüz clear_3d_labels yok)
            # Bu yüzden her tick'te yeniden oluşturmak daha güvenli.
            
            if self.last_data["mesh"]:
                scene.add_geometry("stone_mesh", self.last_data["mesh"], mat_stone)

            if self.last_data["obb"]:
                obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(self.last_data["obb"])
                scene.add_geometry("obb", obb_lines, mat_box)

                # --- 3D ETİKETLERİ EKLEME BÖLÜMÜ ---
                dims = self.last_data["measurements"]["dimensions_mm"]
                corners = np.asarray(self.last_data["obb"].get_box_points())
                
                # OBB köşe sıralaması:
                #    4----5
                #   /|   /|
                #  7----6 |
                #  | 0--|-1
                #  |/   |/
                #  3----2

                # Uzunluk (X ekseni boyunca - en uzun kenar)
                edge_midpoint_L = (corners[0] + corners[1]) / 2.0
                label_text_L = f"{dims[0] / 10.0:.1f} cm"
                scene.add_3d_label(edge_midpoint_L, label_text_L)

                # Genişlik (Y ekseni boyunca)
                edge_midpoint_W = (corners[0] + corners[3]) / 2.0
                label_text_W = f"{dims[1] / 10.0:.1f} cm"
                scene.add_3d_label(edge_midpoint_W, label_text_W)
                
                # Yükseklik (Z ekseni boyunca)
                edge_midpoint_H = (corners[0] + corners[4]) / 2.0
                label_text_H = f"{dims[2] / 10.0:.1f} cm"
                scene.add_3d_label(edge_midpoint_H, label_text_H)


        def on_tick_event():
            """Her frame'de çalışan ana döngü."""
            if self.mode == "realtime":
                points_np, image_bgr = self.get_point_cloud_and_image()
                if points_np is None:
                    return

                _, stone_pcd = self.process_point_cloud(points_np)
                self.last_data["measurements"] = self.calculate_measurements(stone_pcd)
                
                if self.last_data["measurements"]:
                    self.last_data["mesh"] = self.create_stone_mesh(stone_pcd)
                    self.last_data["obb"] = self.last_data["measurements"]["oriented_bounding_box"]
                else:
                    self.last_data["mesh"] = None
                    self.last_data["obb"] = None
                
                self.last_data["image"] = image_bgr.copy()
                
                # Sahneyi güncellemek için ana GUI thread'ine postala
                gui.Application.instance.post_to_main_thread(w, update_scene)

            # CV2 Penceresini güncelle
            display_image = self.last_data["image"].copy()
            if self.last_data["measurements"]:
                m = self.last_data["measurements"]
                cv2.putText(display_image, f"Boyutlar (cm): {m['dimensions_mm'][0]/10:.1f}x{m['dimensions_mm'][1]/10:.1f}x{m['dimensions_mm'][2]/10:.1f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, f"Hacim: {m['volume_cm3']:.1f} cm3", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_image, f"Tahmini Agirlik: {m['weight_kg']:.2f} kg", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_image, f"Mod: {self.mode.upper()}", (display_image.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Kamera ve Sonuclar", display_image)
            k = cv2.waitKey(1)
            if k == ord('q'):
                gui.Application.instance.quit()
            elif k == ord('s'):
                self.mode = "snapshot"
                print("📸 Snapshot moduna geçildi. 3D sahnede gezinebilirsiniz. 'R' ile devam edin.")
            elif k == ord('r'):
                self.mode = "realtime"
                print("🎥 Realtime moda geçildi.")
            elif k == ord('o'):
                if self.last_data["mesh"]:
                    fname = f"stone_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
                    o3d.io.write_triangle_mesh(fname, self.last_data["mesh"])
                    print(f"💾 3D model OBJ olarak kaydedildi: {fname}")

        # Zamanlayıcıyı ayarla (tick event)
        gui.Application.instance.set_on_tick_event(on_tick_event)
        
        # Kamerayı ayarla
        bounds = o3d.geometry.AxisAlignedBoundingBox([-2000, -2000, -2000], [2000, 2000, 2000])
        scene_widget.setup_camera(60, bounds, bounds.get_center())

        # GUI uygulamasını çalıştır
        gui.Application.instance.run()

        # Temizlik
        cv2.destroyAllWindows()
        self.zed.close()


if __name__ == "__main__":
    detector = StoneDimensionEstimator()
    detector.run_customer_demo()