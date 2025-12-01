# zed_packing_with_calib_OBB_aligned.py
import sys
import os
import datetime
import traceback
import math

import numpy as np
import cv2
import open3d as o3d
import pyzed.sl as sl

try:
    import trimesh
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False


class StoneDimensionEstimator:
    def __init__(self,
                 voxel_size_mm=20.0,
                 target_dims_cm=(30.0, 20.0, 10.0),
                 allow_rotations=True,
                 max_voxels=60_000_000):
        self.voxel_size_mm = float(voxel_size_mm)
        self.target_dims_cm = tuple(target_dims_cm)
        self.allow_rotations = bool(allow_rotations)
        self.max_voxels = int(max_voxels)

        self.scale_factor = 1.0

        # ZED init
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
        print("Kamera intrinsic parametreleri alındı.")
        print(f"Voxel: {self.voxel_size_mm} mm | Target (cm): {self.target_dims_cm} | Rot: {self.allow_rotations}")

        # Kalibrasyon için
        self.last_xyz_map = None
        self.last_bgr = None
        self.calib_clicks = []
        
        # Yakalama / analiz durumu
        self.captured_data = None
        self.analysis_results = None

    # ---------- ZED helpers ----------
    def get_point_cloud_and_image(self):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            pc_raw = self.point_cloud.get_data().astype(np.float32)
            xyz_raw = pc_raw[:, :, :3]
            xyz_scaled = xyz_raw * self.scale_factor
            self.last_xyz_map = xyz_scaled.copy()

            mask = np.isfinite(xyz_scaled).all(axis=2)
            valid_points = xyz_scaled[mask]

            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            img = self.image.get_data()
            try:
                image_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            except Exception:
                image_bgr = img[..., :3][:, :, ::-1]
            self.last_bgr = image_bgr.copy()
            return valid_points, image_bgr
        return None, None

    def process_point_cloud(self, points_np, voxel_size=5.0):
        if points_np is None or len(points_np) == 0:
            return None, None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        if len(pcd_down.points) == 0:
            return pcd_down, None
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        try:
            _, inliers = pcd_clean.segment_plane(distance_threshold=15.0, ransac_n=3, num_iterations=1000)
            objects_pcd = pcd_clean.select_by_index(inliers, invert=True)
        except Exception:
            objects_pcd = pcd_clean
        if len(objects_pcd.points) == 0:
            return pcd_clean, None
        labels = np.array(objects_pcd.cluster_dbscan(eps=25.0, min_points=25, print_progress=False))
        if labels.size == 0 or labels.max() == -1:
            return pcd_clean, None
        counts = np.bincount(labels[labels >= 0])
        if counts.size == 0:
            return pcd_clean, None
        largest = np.argmax(counts)
        idx = np.where(labels == largest)[0]
        stone_pcd = objects_pcd.select_by_index(idx)
        return pcd_clean, stone_pcd

    def calculate_measurements(self, stone_pcd):
        if stone_pcd is None or len(stone_pcd.points) < 10:
            return None
        obb = stone_pcd.get_oriented_bounding_box()
        obb.color = (1.0, 0.0, 0.0)
        dimensions = np.asarray(obb.extent)
        volume_mm3 = 0.0
        try:
            hull, _ = stone_pcd.compute_convex_hull()
            if hasattr(hull, "get_volume"):
                try:
                    volume_mm3 = float(hull.get_volume())
                except Exception:
                    volume_mm3 = 0.0
        except Exception:
            volume_mm3 = 0.0
        volume_cm3 = volume_mm3 / 1000.0
        stone_density_g_cm3 = 2.7
        weight_kg = (volume_cm3 * stone_density_g_cm3) / 1000.0
        return {"dimensions_mm": dimensions, "volume_cm3": volume_cm3, "weight_kg": weight_kg, "oriented_bounding_box": obb}

    def create_stone_mesh(self, stone_pcd, alpha=25.0):
        if stone_pcd is None or len(stone_pcd.points) < 20:
            return None
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(stone_pcd, alpha)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.65, 0.65, 0.65])
            return mesh
        except Exception:
            try:
                hull, _ = stone_pcd.compute_convex_hull()
                hull.compute_vertex_normals()
                hull.paint_uniform_color([0.65, 0.65, 0.65])
                return hull
            except Exception:
                return None

    # ---------- OBB'ye hizalı grid yerleştirme ----------
    @staticmethod
    def _mesh_transform(mesh, R, t):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        mesh2 = o3d.geometry.TriangleMesh(mesh)
        mesh2.transform(T)
        return mesh2

    def _pack_in_obb_grid(self, obb, target_dims_cm):
        """
        OBB içine hedef boyutta (cm) dikdörtgen prizmaları ızgara şeklinde dizer.
        Rotasyon serbestse 6 permütasyonu dener; en çok adet çıkaranı seçer.
        Dönüş: dict with 'placed_boxes_world', 'placed_count', 'efficiency', 'piece_mm', 'grid_counts'
        """
        ex, ey, ez = obb.extent  # mm
        perms = [(0,1,2)]
        if self.allow_rotations:
            perms = [
                (0,1,2),(0,2,1),
                (1,0,2),(1,2,0),
                (2,0,1),(2,1,0)
            ]

        best = None
        target_mm = np.array(target_dims_cm, dtype=float) * 10.0  # (L,W,H) mm

        for p in perms:
            piece_mm = target_mm[list(p)]  # bu permütasyonda (x,y,z)
            nx = int(ex // piece_mm[0])
            ny = int(ey // piece_mm[1])
            nz = int(ez // piece_mm[2])
            count = nx * ny * nz
            if best is None or count > best["count"]:
                best = {"perm": p, "piece_mm": piece_mm, "grid": (nx, ny, nz), "count": count}

        piece_mm = best["piece_mm"]
        nx, ny, nz = best["grid"]
        placed = []
        R = obb.R
        c = np.asarray(obb.center)

        # OBB lokalinde (-ext/2 .. +ext/2)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    w, h, d = piece_mm[0], piece_mm[1], piece_mm[2]
                    # kutu merkezi (lokal)
                    lx = -ex/2 + (ix + 0.5) * w
                    ly = -ey/2 + (iy + 0.5) * h
                    lz = -ez/2 + (iz + 0.5) * d

                    box_local = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
                    box_local.compute_vertex_normals()
                    box_local.paint_uniform_color([0.1, 0.9, 0.1])
                    # create_box lokali (0..w,0..h,0..d) olduğu için merkezi (lx,ly,lz) olacak şekilde kaydır:
                    box_local.translate([lx - w/2, ly - h/2, lz - d/2], relative=True)
                    # dünyaya
                    box_world = self._mesh_transform(box_local, R, c)

                    # min köşe (dünya)
                    min_local = np.array([lx - w/2, ly - h/2, lz - d/2])
                    min_world = (R @ min_local) + c
                    center_local = np.array([lx, ly, lz])
                    center_world = (R @ center_local) + c

                    placed.append({
                        "mesh": box_world,
                        "csv_data": (min_world[0], min_world[1], min_world[2], w, h, d, center_world)
                    })

        # verim hesabı: konan toplam parça hacmi / taş hacmi (mm^3)
        stone_vol_cm3_est = None
        efficiency = 0.0
        return {
            "placed_boxes_world": placed,
            "placed_count": len(placed),
            "piece_mm": piece_mm,
            "grid_counts": (nx, ny, nz),
            "efficiency": efficiency  # gerçek taş hacmine göre aşağıda dolduruyoruz
        }

    def _project_3d_to_pixel(self, p3):
        fx, fy, cx, cy = self.camera_intrinsics.fx, self.camera_intrinsics.fy, self.camera_intrinsics.cx, self.camera_intrinsics.cy
        x, y, z = float(p3[0]), float(p3[1]), float(p3[2])
        if z <= 0 or not np.isfinite(z): return None
        u = int((x * fx / z) + cx); v = int((y * fy / z) + cy)
        return (u, v)

    # ---------- Ağır analizi yapan yardımcı ----------
    def _perform_full_analysis(self, points_np):
        print("1/3 - Nokta bulutu işleniyor...")
        _, stone_pcd = self.process_point_cloud(points_np)

        print("2/3 - Ölçümler hesaplanıyor...")
        measurements = self.calculate_measurements(stone_pcd)

        print("3/3 - Mesh oluşturuluyor ve OBB-Grid yerleşim yapılıyor...")
        stone_mesh = self.create_stone_mesh(stone_pcd)

        results = {
            "stone_pcd": stone_pcd,
            "measurements": measurements,
            "stone_mesh": stone_mesh,
            "placed_boxes_world": [], "placed_count": 0,
            "theoretical_by_volume": 0, "efficiency": 0.0, "total_vol_cm3": 0.0
        }

        if stone_mesh is not None and measurements is not None:
            try:
                obb = measurements["oriented_bounding_box"]

                # OBB içine ızgara yerleşim
                pack = self._pack_in_obb_grid(obb, self.target_dims_cm)

                results["placed_boxes_world"] = pack["placed_boxes_world"]
                results["placed_count"] = pack["placed_count"]

                # hacim-tabanlı teorik/efficiency
                # taş hacmi ~ konveks-hacim (calculate_measurements içinde cm^3 var)
                total_vol_cm3 = measurements["volume_cm3"]
                results["total_vol_cm3"] = total_vol_cm3
                piece_vol_mm3 = float(np.prod(pack["piece_mm"]))
                piece_vol_cm3 = piece_vol_mm3 / 1000.0
                if piece_vol_cm3 > 0:
                    results["theoretical_by_volume"] = int(math.floor((total_vol_cm3) / piece_vol_cm3))
                # verim = yerleşen parça hacmi / taş hacmi
                placed_volume_cm3 = results["placed_count"] * piece_vol_cm3
                if total_vol_cm3 > 1e-6:
                    results["efficiency"] = 100.0 * (placed_volume_cm3 / total_vol_cm3)
            except Exception:
                traceback.print_exc()

        print("Analiz tamamlandı.")
        return results

    # ---------- Calibration ----------
    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.last_xyz_map is None: print("XYZ haritası yok."); return
            self.calib_clicks.append((x, y))
            print(f"Kalibrasyon tıkı: {(x, y)}")
            if len(self.calib_clicks) == 2:
                try:
                    p1 = self._pixel_to_3d(self.calib_clicks[0])
                    p2 = self._pixel_to_3d(self.calib_clicks[1])
                    if p1 is None or p2 is None:
                        print("Geçersiz derinlik okuması. Tekrar deneyin (F).")
                    else:
                        d_meas = np.linalg.norm(p1 - p2)
                        print(f"Ölçülen mesafe: {d_meas:.2f} mm")
                        try:
                            true_mm = float(input("Gerçek mesafeyi mm olarak girin: ").strip())
                            if true_mm > 0:
                                sf = true_mm / max(d_meas, 1e-6)
                                self.scale_factor *= sf
                                print(f"Yeni scale_factor = {self.scale_factor:.6f}")
                            else: print("Geçersiz değer, kalibrasyon iptal.")
                        except Exception: print("Girdi hatası, kalibrasyon iptal.")
                finally:
                    self.calib_clicks.clear()
                    cv2.setMouseCallback("Kamera Görüntüsü", lambda *args: None)

    def _pixel_to_3d(self, xy, k=2):
        if self.last_xyz_map is None: return None
        h, w, _ = self.last_xyz_map.shape; x, y = int(xy[0]), int(xy[1])
        xmin, xmax = max(0, x-k), min(w-1, x+k); ymin, ymax = max(0, y-k), min(h-1, y+k)
        patch = self.last_xyz_map[ymin:ymax+1, xmin:xmax+1, :]; valid = np.isfinite(patch).all(axis=2)
        if not np.any(valid): return None
        return np.median(patch[valid], axis=0)

    # ---------- Main loop ----------
    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("3D Analiz", width=1100, height=700, left=50, top=50)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.18])
        opt.mesh_show_back_face = True
        
        print("\n--- KULLANIM ---")
        print(" C: Anlık görüntüyü yakala ve analizi başlat")
        print(" R: Analizi sıfırla ve canlı görüntüye dön")
        print(" F: Referans ile kalibrasyon (2 tıkla)")
        print(" K: Kalibrasyon faktörünü sıfırla (1.0 yap)")
        print(" E: CSV olarak dışa aktar (analiz sonrası)")
        print(" Q: Çıkış")
        print("----------------\n")

        should_exit = {"v": False}
        def cb_q(_): should_exit["v"] = True; return False
        vis.register_key_callback(ord("Q"), cb_q)

        try:
            while not should_exit["v"]:
                # --- CANLI GÖRÜNTÜ MODU ---
                if self.captured_data is None:
                    points_np, image_bgr = self.get_point_cloud_and_image()
                    if image_bgr is None:
                        continue
                    
                    h, w, _ = image_bgr.shape
                    cv2.putText(image_bgr, "CANLI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image_bgr, "Analiz icin 'C' tusuna basin", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    for (cx, cy) in self.calib_clicks:
                        cv2.circle(image_bgr, (cx, cy), 6, (0, 0, 255), -1)

                    cv2.imshow("Kamera Görüntüsü", image_bgr)
                    
                    vis.clear_geometries()
                    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=100))
                    vis.poll_events()
                    vis.update_renderer()

                # --- ANALİZ MODU ---
                else: 
                    if self.analysis_results is None:
                        img_wait = self.captured_data['image'].copy()
                        h, w, _ = img_wait.shape
                        cv2.putText(img_wait, "ANALIZ EDILIYOR...", (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.imshow("Kamera Görüntüsü", img_wait)
                        cv2.waitKey(1)
                        
                        self.analysis_results = self._perform_full_analysis(self.captured_data['points'])

                        vis.clear_geometries()
                        if self.analysis_results['stone_mesh']:
                            vis.add_geometry(self.analysis_results['stone_mesh'])
                        if self.analysis_results['measurements']:
                            vis.add_geometry(self.analysis_results['measurements']['oriented_bounding_box'])
                        for box_data in self.analysis_results['placed_boxes_world']:
                            vis.add_geometry(box_data['mesh'])

                    image_display = self.captured_data['image'].copy()
                    h, w = image_display.shape[:2]
                    panel_h = 130
                    cv2.rectangle(image_display, (0, 0), (w, panel_h), (0, 0, 0), -1)
                    
                    y = 25
                    res = self.analysis_results
                    if res['measurements']:
                        dims = res['measurements']["dimensions_mm"]
                        cv2.putText(image_display, f"Tas OBB (cm): {dims[0]/10:.1f}x{dims[1]/10:.1f}x{dims[2]/10:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 22
                    cv2.putText(image_display, f"Hedef (cm): {self.target_dims_cm[0]:.1f}x{self.target_dims_cm[1]:.1f}x{self.target_dims_cm[2]:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1); y += 20
                    if res['total_vol_cm3'] > 0:
                        cv2.putText(image_display, f"Hacim~ (cm3): {res['total_vol_cm3']:.0f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); y += 22
                    cv2.putText(image_display, f"Yerlesen: {res['placed_count']} adet | Teorik: {res['theoretical_by_volume']} | Verim: {res['efficiency']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2); y += 22

                    # Kutu indekslerini görüntüye yaz
                    for idx, box_data in enumerate(res['placed_boxes_world'], start=1):
                        pt = self._project_3d_to_pixel(box_data['csv_data'][6]) # center_world
                        if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                            cv2.circle(image_display, pt, 4, (0, 0, 255), -1)
                            cv2.putText(image_display, f"{idx}", (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    cv2.putText(image_display, "Sifirlamak icin 'R' tusuna basin", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Kamera Görüntüsü", image_display)
                    vis.poll_events()
                    vis.update_renderer()

                # --- TUŞ KONTROLLERİ ---
                k = cv2.waitKey(10) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('c') and self.captured_data is None:
                    print("\nGoruntu yakalandi, analiz baslatiliyor...")
                    # canlı döngüdeki son okuma değişkenleri
                    points_np, image_bgr = self.get_point_cloud_and_image()
                    self.captured_data = {'points': points_np, 'image': image_bgr}
                elif k == ord('r'):
                    print("\nAnaliz sifirlandi, canli goruntuye donuluyor.")
                    self.captured_data = None
                    self.analysis_results = None
                elif k == ord('f'):
                    self.calib_clicks.clear()
                    print("Kalibrasyon: 2 noktaya tiklayin, sonra konsola gercek mm girin.")
                    cv2.setMouseCallback("Kamera Görüntüsü", self._mouse_cb)
                elif k == ord('k'):
                    self.scale_factor = 1.0
                    print(f"Kalibrasyon faktoru sifirlandi: {self.scale_factor}")
                elif k == ord('e'):
                    if self.analysis_results and self.analysis_results['placed_boxes_world']:
                        fname = f"cut_plan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        try:
                            with open(fname, "w", encoding="utf-8") as f:
                                f.write("min_x_mm,min_y_mm,min_z_mm,size_x_mm,size_y_mm,size_z_mm,center_x_mm,center_y_mm,center_z_mm\n")
                                for box in self.analysis_results['placed_boxes_world']:
                                    vx,vy,vz,sx,sy,sz,cc = box['csv_data']
                                    f.write(f"{vx:.2f},{vy:.2f},{vz:.2f},{sx:.2f},{sy:.2f},{sz:.2f},{cc[0]:.2f},{cc[1]:.2f},{cc[2]:.2f}\n")
                            print(f"Kesim plani CSV kaydedildi: {fname}")
                        except Exception: traceback.print_exc()
                    else:
                        print("CSV'ye aktarilacak analiz sonucu bulunamadi.")
        
        finally:
            print("Kapanıyor...")
            vis.destroy_window()
            cv2.destroyAllWindows()
            self.zed.close()

if __name__ == "__main__":
    detector = StoneDimensionEstimator(
        voxel_size_mm=20.0,                 # bu modda doğrudan kullanılmıyor ama dursun
        target_dims_cm=(30.0, 20.0, 10.0),  # istediğin parça ölçüsü (cm)
        allow_rotations=True                # 6 permütasyonu dene
    )
    detector.run()
