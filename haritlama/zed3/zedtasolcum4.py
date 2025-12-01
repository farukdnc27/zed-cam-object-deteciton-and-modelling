import sys
import os
import datetime
import traceback

import numpy as np
import cv2
import open3d as o3d
import pyzed.sl as sl


class StoneDimensionEstimator:
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
        """ZED'den nokta bulutu (Nx3 numpy in mm) ve BGR görüntü döndürür."""
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            pc = self.point_cloud.get_data()  # H x W x 4
            points = pc[:, :, :3].astype(np.float32)
            mask = np.isfinite(points).all(axis=2)
            valid_points = points[mask]  # (N,3) in millimeters

            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            img = self.image.get_data()  # RGBA
            # RGBA -> BGR
            try:
                image_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            except Exception:
                image_bgr = img[..., :3][:, :, ::-1]
            return valid_points, image_bgr
        return None, None

    def process_point_cloud(self, points_np, voxel_size=5.0):
        """Open3D PointCloud oluşturur, zemini segment edip en büyük cluster'ı taş olarak döndürür."""
        if points_np is None or len(points_np) == 0:
            return None, None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        # Voxel downsample (mm biriminde voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        if len(pcd_down.points) == 0:
            return pcd_down, None

        # Temizleme
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Zemin düzlemi segmentasyonu
        try:
            plane_model, inliers = pcd_clean.segment_plane(distance_threshold=15.0, ransac_n=3, num_iterations=1000)
            objects_pcd = pcd_clean.select_by_index(inliers, invert=True)
        except Exception:
            objects_pcd = pcd_clean

        if len(objects_pcd.points) == 0:
            return pcd_clean, None

        # Cluster - en büyük cluster'ı al
        labels = np.array(objects_pcd.cluster_dbscan(eps=25.0, min_points=25, print_progress=False))
        if labels.size == 0 or labels.max() == -1:
            return pcd_clean, None

        counts = np.bincount(labels[labels >= 0])
        if counts.size == 0:
            return pcd_clean, None

        largest_cluster_label = np.argmax(counts)
        stone_indices = np.where(labels == largest_cluster_label)[0]
        stone_pcd = objects_pcd.select_by_index(stone_indices)
        return pcd_clean, stone_pcd

    def calculate_measurements(self, stone_pcd):
        """OBB, boyutlar, hacim (cm3) ve ağırlık (kg) hesaplar."""
        if stone_pcd is None or len(stone_pcd.points) < 10:
            return None

        obb = stone_pcd.get_oriented_bounding_box()
        obb.color = (1.0, 0.0, 0.0)
        dimensions = np.asarray(obb.extent)  # mm

        volume_mm3 = 0.0
        try:
            hull, _ = stone_pcd.compute_convex_hull()
            # hull.get_volume() döndürür (aynı birimde kübik)
            if hasattr(hull, "is_watertight") and hull.is_watertight():
                volume_mm3 = hull.get_volume()
            else:
                # yine de bir hacim dönebilir; kullan
                volume_mm3 = hull.get_volume() if hasattr(hull, "get_volume") else 0.0
        except Exception:
            volume_mm3 = 0.0

        # mm^3 -> cm^3 (1 cm^3 = 1000 mm^3)
        volume_cm3 = float(volume_mm3) / 1000.0
        stone_density_g_cm3 = 2.7  # granit-benzeri; istersen değiştir
        weight_kg = (volume_cm3 * stone_density_g_cm3) / 1000.0

        return {
            "dimensions_mm": dimensions,
            "volume_cm3": volume_cm3,
            "weight_kg": weight_kg,
            "oriented_bounding_box": obb,
        }

    def create_stone_mesh(self, stone_pcd, alpha=25.0):
        """Alpha-shape ile mesh üret, başarısız olursa convex hull'ı döndür."""
        if stone_pcd is None or len(stone_pcd.points) < 20:
            return None
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(stone_pcd, alpha)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
            return mesh
        except Exception:
            try:
                hull, _ = stone_pcd.compute_convex_hull()
                hull.compute_vertex_normals()
                hull.paint_uniform_color([0.7, 0.7, 0.7])
                return hull
            except Exception:
                return None

    def project_3d_to_2d(self, point_3d):
        fx = self.camera_intrinsics.fx
        fy = self.camera_intrinsics.fy
        cx = self.camera_intrinsics.cx
        cy = self.camera_intrinsics.cy
        x, y, z = point_3d
        if z == 0:
            return None
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        return (u, v)

    def run_customer_demo(self):
        # Open3D görselleştiriciyi KeyCallback destekli şekilde açıyoruz
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("3D Analiz", width=960, height=720, left=50, top=50)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.18])
        opt.mesh_show_back_face = True
        view_control = vis.get_view_control()

        # Mod ve durum tutucular
        mode = {"value": "realtime"}  # 'realtime' veya 'snapshot'
        last_mesh = {"mesh": None}
        last_obb = {"obb": None}
        last_image = {"img": None}
        snapshot_prepared = {"val": False}
        should_exit = {"val": False}

        # Klasik tuşlarla (Open3D penceresine basınca çalışır):
        def cb_r(vis_obj):
            mode["value"] = "realtime"
            snapshot_prepared["val"] = False
            print("🎥 Realtime moda geçildi (Open3D tuş).")
            return False

        def cb_s(vis_obj):
            # snapshot'a geçiş: önceden en son yakalanan mesh/obb yoksa uyar
            if last_mesh["mesh"] is None and last_obb["obb"] is None:
                print("⚠️ Snapshot için henüz veri yok.")
                return False
            mode["value"] = "snapshot"
            # Sadece ilk kez snapshot'a geçerken geometrileri sabitle
            if not snapshot_prepared["val"]:
                try:
                    prev_cam = view_control.convert_to_pinhole_camera_parameters()
                except Exception:
                    prev_cam = None
                vis.clear_geometries()
                if last_mesh["mesh"] is not None:
                    vis.add_geometry(last_mesh["mesh"])
                if last_obb["obb"] is not None:
                    vis.add_geometry(last_obb["obb"])
                if prev_cam:
                    try:
                        view_control.convert_from_pinhole_camera_parameters(prev_cam)
                    except Exception:
                        pass
                snapshot_prepared["val"] = True
            print("📸 Snapshot moda geçildi (Open3D tuş). Fare ile zoom/pan/rotate yapabilirsiniz.")
            return False

        def cb_k(vis_obj):
            # Kamera görüntüsü kaydet
            if mode["value"] == "snapshot" and last_image["img"] is not None:
                filename = f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, last_image["img"])
                print(f"💾 Kamera görüntüsü kaydedildi: {filename}")
            else:
                print("⚠️ Snapshot modunda değilsiniz veya görüntü yok.")
            return False

        def cb_o(vis_obj):
            # OBJ export
            if last_mesh["mesh"] is None:
                print("⚠️ Export için mesh yok.")
                return False
            fname = f"stone_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
            o3d.io.write_triangle_mesh(fname, last_mesh["mesh"], write_ascii=True)
            print(f"💾 3D mesh OBJ olarak kaydedildi: {fname}")
            return False

        def cb_p(vis_obj):
            # PLY export
            if last_mesh["mesh"] is None:
                print("⚠️ Export için mesh yok.")
                return False
            fname = f"stone_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
            o3d.io.write_triangle_mesh(fname, last_mesh["mesh"])
            print(f"💾 3D mesh PLY olarak kaydedildi: {fname}")
            return False

        def cb_q(vis_obj):
            should_exit["val"] = True
            print("Çıkış tetiklendi (Open3D tuş).")
            return False

        # Kayıt
        vis.register_key_callback(ord("R"), cb_r)
        vis.register_key_callback(ord("S"), cb_s)
        vis.register_key_callback(ord("K"), cb_k)
        vis.register_key_callback(ord("O"), cb_o)
        vis.register_key_callback(ord("P"), cb_p)
        vis.register_key_callback(ord("Q"), cb_q)

        # Ayrıca cv2 penceresinden de tuş yakalayabilirsin (bazen Open3D penceresi odakta olmaz)
        print("Klavye kısayolları: R=realtime, S=snapshot, K=save image, O=export OBJ, P=export PLY, Q=quit")
        try:
            while not should_exit["val"]:
                # cv2 tuşları (bedeni)
                k = cv2.waitKey(1) & 0xFF
                if k != 255:
                    if k == ord("q"):
                        print("Çıkış tuşu (cv2).")
                        break
                    elif k == ord("r"):
                        cb_r(vis)
                    elif k == ord("s"):
                        cb_s(vis)
                    elif k == ord("k"):
                        cb_k(vis)
                    elif k == ord("o"):
                        cb_o(vis)
                    elif k == ord("p"):
                        cb_p(vis)

                if mode["value"] == "realtime":
                    # Gerçek zamanlı mod: ZED'den al, işle, sahneyi güncelle
                    try:
                        points_np, image_bgr = self.get_point_cloud_and_image()
                        if points_np is None or len(points_np) < 50:
                            # çok az veri -> atla
                            cv2.imshow("Kamera Görüntüsü", np.zeros((480, 640, 3), dtype=np.uint8))
                            vis.poll_events()
                            vis.update_renderer()
                            continue

                        _, stone_pcd = self.process_point_cloud(points_np)
                        measurements = self.calculate_measurements(stone_pcd)
                        stone_mesh = self.create_stone_mesh(stone_pcd)

                        # Son durumu sakla (export ve snapshot için)
                        last_mesh["mesh"] = stone_mesh
                        last_obb["obb"] = measurements["oriented_bounding_box"] if measurements else None
                        last_image["img"] = image_bgr.copy()

                        # Sahneyi güncelle (kamera pozisyonunu koru)
                        try:
                            prev_cam = view_control.convert_to_pinhole_camera_parameters()
                        except Exception:
                            prev_cam = None

                        vis.clear_geometries()
                        if stone_mesh is not None:
                            vis.add_geometry(stone_mesh)
                        if measurements is not None:
                            vis.add_geometry(measurements["oriented_bounding_box"])

                        if prev_cam is not None:
                            try:
                                view_control.convert_from_pinhole_camera_parameters(prev_cam)
                            except Exception:
                                pass

                        snapshot_prepared["val"] = False  # realtime da snapshot artık yeniden hazırlansın

                        vis.poll_events()
                        vis.update_renderer()

                        # 2D overlay metinleri
                        if measurements is not None:
                            dims = measurements["dimensions_mm"]
                            dim_text = f"{dims[0]/10:.1f} x {dims[1]/10:.1f} x {dims[2]/10:.1f} cm"
                            volume_text = f"Hacim: {measurements['volume_cm3']:.0f} cm³"
                            weight_text = f"Ağırlık: {measurements['weight_kg']:.2f} kg"
                            cv2.putText(image_bgr, dim_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(image_bgr, volume_text, (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv2.putText(image_bgr, weight_text, (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            # merkez işareti
                            center_3d = measurements["oriented_bounding_box"].get_center()
                            center_2d = self.project_3d_to_2d(center_3d)
                            if center_2d:
                                cv2.circle(image_bgr, center_2d, 6, (0, 0, 255), -1)

                        cv2.imshow("Kamera Görüntüsü", image_bgr)

                    except Exception:
                        traceback.print_exc()
                        # hata olsa da döngü devam eder

                else:
                    # snapshot modu: ZED'den yeni veri çekme, ekleme/çıkarma yapma
                    # sadece Open3D penceresini güncelle ve snapshot görüntüsünü göster
                    vis.poll_events()
                    vis.update_renderer()
                    if last_image["img"] is not None:
                        cv2.imshow("Kamera Görüntüsü", last_image["img"])
                    else:
                        cv2.imshow("Kamera Görüntüsü", np.zeros((480, 640, 3), dtype=np.uint8))

        finally:
            print("Kapanıyor...")
            vis.destroy_window()
            cv2.destroyAllWindows()
            self.zed.close()


if __name__ == "__main__":
    detector = StoneDimensionEstimator()
    detector.run_customer_demo()
