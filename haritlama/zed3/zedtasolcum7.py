# zed_packing_with_calib.py
import sys
import os
import datetime
import traceback
import math

import numpy as np
import cv2
import open3d as o3d
import pyzed.sl as sl

# (opsiyonel) daha sağlam nokta-içi testi için
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
        """
        voxel_size_mm: voxel çözünürlüğü (mm)
        target_dims_cm: hedef parça ölçüsü (cm) (L, W, H)
        allow_rotations: 90° eksen rotasyonlarına izin ver
        max_voxels: bellek koruması
        """
        self.voxel_size_mm = float(voxel_size_mm)
        self.target_dims_cm = tuple(target_dims_cm)
        self.allow_rotations = bool(allow_rotations)
        self.max_voxels = int(max_voxels)

        # referans ölçek düzeltmesi (F ile değişir)
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

        # kalibrasyon için son RGB ve XYZ haritasını sakla
        self.last_xyz_map = None   # HxWx3 (mm)
        self.last_bgr = None
        self.calib_clicks = []     # [(x,y), (x,y)]

    # ---------- ZED helpers ----------
    def get_point_cloud_and_image(self):
        """
        Dönüş:
          valid_points (N,3)  -> işlemede hız için (ölçek uygulanmış)
          image_bgr (H,W,3)
        Ek olarak self.last_xyz_map (H,W,3) NaN'lı şekilde güncellenir (ölçek uygulanmış)
        """
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            pc = self.point_cloud.get_data().astype(np.float32)  # H x W x 4
            xyz = pc[:, :, :3] * self.scale_factor  # ölçek uygula

            # XYZ haritasını sakla (kalibrasyon için)
            self.last_xyz_map = xyz.copy()

            mask = np.isfinite(xyz).all(axis=2)
            valid_points = xyz[mask]

            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            img = self.image.get_data()  # RGBA
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
        dimensions = np.asarray(obb.extent)  # mm
        # hızlı özet için konveks hacim
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
        return {
            "dimensions_mm": dimensions,
            "volume_cm3": volume_cm3,
            "weight_kg": weight_kg,
            "oriented_bounding_box": obb,
        }

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

    # ---------- voxelization & packing ----------
    def voxelize_mesh_to_bool(self, mesh, voxel_size_mm):
        """
        grid (nz,ny,nx) boolean: True=taşın içinde/uygun
        origin: AABB min bound (mm)
        """
        aabb = mesh.get_axis_aligned_bounding_box()
        minb = np.asarray(aabb.min_bound)
        maxb = np.asarray(aabb.max_bound)
        dims = (maxb - minb)
        nx = int(np.ceil(dims[0] / voxel_size_mm))
        ny = int(np.ceil(dims[1] / voxel_size_mm))
        nz = int(np.ceil(dims[2] / voxel_size_mm))
        total = nx * ny * nz
        if total > self.max_voxels:
            raise MemoryError(f"Voxel grid çok büyük: {total}. voxel_size_mm'i büyüt.")
        grid = np.zeros((nz, ny, nx), dtype=bool)

        xs = (np.arange(nx) + 0.5) * voxel_size_mm + minb[0]
        ys = (np.arange(ny) + 0.5) * voxel_size_mm + minb[1]
        zs = (np.arange(nz) + 0.5) * voxel_size_mm + minb[2]

        if _HAS_TRIMESH:
            tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                 faces=np.asarray(mesh.triangles),
                                 process=False)
            for iz, z in enumerate(zs):
                yy, xx = np.meshgrid(ys, xs, indexing='ij')
                pts = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
                try:
                    inside = tm.contains(pts)
                except Exception:
                    inside = np.zeros(pts.shape[0], dtype=bool)
                grid[iz, :, :] = inside.reshape((ny, nx))
        else:
            sampled = mesh.sample_points_uniformly(number_of_points=200000)
            sp = np.asarray(sampled.points)
            if sp.shape[0] == 0:
                return grid, minb, (nx, ny, nz)
            # kaba yaklaşım: yüzeye yakınlığı eşik al
            try:
                from sklearn.neighbors import KDTree
                tree = KDTree(sp)
                diag = np.sqrt(3 * (voxel_size_mm ** 2)) / 2.0
                for iz, z in enumerate(zs):
                    yy, xx = np.meshgrid(ys, xs, indexing='ij')
                    centers = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
                    d, _ = tree.query(centers, k=1)
                    inside = (d.ravel() < diag * 1.15)
                    grid[iz, :, :] = inside.reshape((ny, nx))
            except Exception:
                # çok yavaş fallback: işaretleme yok
                pass

        return grid, minb, (nx, ny, nz)

    def greedy_pack(self, grid, piece_vox_shape):
        """
        grid: (nz,ny,nx) True=uygun
        piece_vox_shape: (pz,py,px)
        return: placed [(ix,iy,iz, sx,sy,sz)], used_vox, filled(bool)
        """
        nz, ny, nx = grid.shape
        pz0, py0, px0 = piece_vox_shape

        # rotasyon varyantları
        if self.allow_rotations:
            from itertools import permutations
            rotations = list({tuple(x) for x in permutations((pz0, py0, px0))})
        else:
            rotations = [(pz0, py0, px0)]

        filled = np.zeros_like(grid, dtype=bool)
        placed = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    if not grid[iz, iy, ix] or filled[iz, iy, ix]:
                        continue
                    placed_here = False
                    for (pz, py, px) in rotations:
                        iz2, iy2, ix2 = iz + pz, iy + py, ix + px
                        if iz2 > nz or iy2 > ny or ix2 > nx:
                            continue
                        region_ok = np.all(grid[iz:iz2, iy:iy2, ix:ix2])
                        region_free = not np.any(filled[iz:iz2, iy:iy2, ix:ix2])
                        if region_ok and region_free:
                            filled[iz:iz2, iy:iy2, ix:ix2] = True
                            placed.append((ix, iy, iz, px, py, pz))
                            placed_here = True
                            break
                    if placed_here:
                        continue
        used_vox = int(np.count_nonzero(filled))
        return placed, used_vox, filled

    # ---------- calibration (F) ----------
    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.last_xyz_map is None:
                print("XYZ haritası yok.")
                return
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
                            else:
                                print("Geçersiz değer, kalibrasyon iptal.")
                        except Exception:
                            print("Girdi hatası, kalibrasyon iptal.")
                finally:
                    self.calib_clicks.clear()
                    # mouse callback'i kapatma: tekrar F'ye kadar aktif kalsın istersen kaldırma
                    cv2.setMouseCallback("Kamera Görüntüsü", lambda *args: None)

    def _pixel_to_3d(self, xy, k=2):
        """piksel (x,y) çevresinde (2k+1)^2 bölgeden medyan XYZ döndürür."""
        if self.last_xyz_map is None:
            return None
        h, w, _ = self.last_xyz_map.shape
        x, y = int(xy[0]), int(xy[1])
        xmin, xmax = max(0, x - k), min(w - 1, x + k)
        ymin, ymax = max(0, y - k), min(h - 1, y + k)
        patch = self.last_xyz_map[ymin:ymax+1, xmin:xmax+1, :]
        valid = np.isfinite(patch).all(axis=2)
        if not np.any(valid):
            return None
        pts = patch[valid]
        return np.median(pts, axis=0)

    # ---------- main loop ----------
    def run_customer_demo(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("3D Analiz", width=1100, height=700, left=50, top=50)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.18])
        opt.mesh_show_back_face = True
        view = vis.get_view_control()

        print("Kısayollar: F=Ref. kalibrasyon (2 tık), E=CSV export, Q=çıkış")

        should_exit = {"v": False}
        placed_boxes_world = []  # [(minx,miny,minz, sx,sy,sz) mm]

        def cb_q(_):
            should_exit["v"] = True
            return False

        vis.register_key_callback(ord("Q"), cb_q)

        try:
            while not should_exit["v"]:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('f'):
                    self.calib_clicks.clear()
                    print("Kalibrasyon modu: OpenCV penceresinde art arda 2 noktaya tıklayın, sonra konsola gerçek mesafeyi (mm) girin.")
                    cv2.setMouseCallback("Kamera Görüntüsü", self._mouse_cb)
                elif k == ord('e'):
                    if placed_boxes_world:
                        fname = f"cut_plan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        try:
                            with open(fname, "w", encoding="utf-8") as f:
                                f.write("min_x_mm,min_y_mm,min_z_mm,size_x_mm,size_y_mm,size_z_mm\n")
                                for (vx, vy, vz, sx, sy, sz) in placed_boxes_world:
                                    f.write(f"{vx:.2f},{vy:.2f},{vz:.2f},{sx:.2f},{sy:.2f},{sz:.2f}\n")
                            print(f"Kesim planı CSV kaydedildi: {fname}")
                        except Exception:
                            traceback.print_exc()
                    else:
                        print("CSV için yerleştirilmiş kutu yok.")

                try:
                    points_np, image_bgr = self.get_point_cloud_and_image()
                    if points_np is None or len(points_np) < 200:
                        cv2.imshow("Kamera Görüntüsü", np.zeros((480, 640, 3), dtype=np.uint8))
                        vis.poll_events(); vis.update_renderer()
                        continue

                    _, stone_pcd = self.process_point_cloud(points_np)
                    measurements = self.calculate_measurements(stone_pcd)
                    stone_mesh = self.create_stone_mesh(stone_pcd)

                    # --- packing ---
                    placed_boxes_world = []
                    placed_count = 0
                    theoretical_by_volume = 0
                    efficiency = 0.0
                    total_vol_cm3 = 0.0

                    # kamera pozu KORU
                    try:
                        prev_cam = view.convert_to_pinhole_camera_parameters()
                    except Exception:
                        prev_cam = None

                    vis.clear_geometries()

                    if stone_mesh is not None:
                        vis.add_geometry(stone_mesh)

                        try:
                            grid, origin, (nx, ny, nz) = self.voxelize_mesh_to_bool(stone_mesh, self.voxel_size_mm)
                            total_inside = int(np.count_nonzero(grid))
                            total_vol_mm3 = total_inside * (self.voxel_size_mm ** 3)
                            total_vol_cm3 = total_vol_mm3 / 1000.0

                            # target -> voxel
                            tx_mm = self.target_dims_cm[0] * 10.0
                            ty_mm = self.target_dims_cm[1] * 10.0
                            tz_mm = self.target_dims_cm[2] * 10.0
                            px = max(1, int(np.ceil(tx_mm / self.voxel_size_mm)))
                            py = max(1, int(np.ceil(ty_mm / self.voxel_size_mm)))
                            pz = max(1, int(np.ceil(tz_mm / self.voxel_size_mm)))

                            placed, used_vox, _ = self.greedy_pack(grid, (pz, py, px))
                            placed_count = len(placed)
                            theoretical_by_volume = int(math.floor(total_vol_mm3 / max((px*py*pz)*(self.voxel_size_mm**3), 1e-6)))
                            efficiency = (used_vox / max(total_inside, 1)) * 100.0

                            # visualize boxes + store world coords
                            aabb = stone_mesh.get_axis_aligned_bounding_box()
                            minb = np.asarray(aabb.min_bound)
                            for (ix, iy, iz, sx, sy, sz) in placed:
                                vx = minb[0] + ix * self.voxel_size_mm
                                vy = minb[1] + iy * self.voxel_size_mm
                                vz = minb[2] + iz * self.voxel_size_mm
                                wx = sx * self.voxel_size_mm
                                wy = sy * self.voxel_size_mm
                                wz = sz * self.voxel_size_mm

                                box = o3d.geometry.TriangleMesh.create_box(width=wx, height=wy, depth=wz)
                                box.compute_vertex_normals()
                                box.paint_uniform_color([0.1, 0.9, 0.1])   # parlak yeşil
                                box.translate((vx, vy, vz))
                                vis.add_geometry(box)

                                placed_boxes_world.append((vx, vy, vz, wx, wy, wz))
                        except MemoryError as me:
                            print("Voxelizasyon çok büyük:", me)
                        except Exception:
                            traceback.print_exc()

                    if measurements is not None:
                        vis.add_geometry(measurements["oriented_bounding_box"])

                    if prev_cam is not None:
                        try:
                            view.convert_from_pinhole_camera_parameters(prev_cam)
                        except Exception:
                            pass

                    vis.poll_events(); vis.update_renderer()

                    # --- 2D overlay ---
                    h, w = image_bgr.shape[:2]
                    panel_h = 120
                    cv2.rectangle(image_bgr, (0, 0), (w, panel_h), (0, 0, 0), -1)

                    y = 25
                    if measurements is not None:
                        dims = measurements["dimensions_mm"]
                        cv2.putText(image_bgr, f"Taş OBB (cm): {dims[0]/10:.1f} x {dims[1]/10:.1f} x {dims[2]/10:.1f}",
                                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2); y += 25

                    cv2.putText(image_bgr, f"Hedef Parca (cm): {self.target_dims_cm[0]:.1f} x {self.target_dims_cm[1]:.1f} x {self.target_dims_cm[2]:.1f}",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += 22

                    if total_vol_cm3 > 0:
                        cv2.putText(image_bgr, f"Hacim~ (cm3): {total_vol_cm3:.0f}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); y += 22

                    cv2.putText(image_bgr, f"Yerleşen Adet: {placed_count}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2); y += 25
                    cv2.putText(image_bgr, f"Teorik (hacme gore): {theoretical_by_volume}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2); y += 22
                    cv2.putText(image_bgr, f"Verim: {efficiency:.1f}%", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2); y += 24

                    cv2.putText(image_bgr, "Kisayol: F=Kalibrasyon, E=CSV, Q=Quit",
                                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

                    # kalibrasyon tıklarını görsel işaretle
                    for (cx, cy) in self.calib_clicks:
                        cv2.circle(image_bgr, (cx, cy), 6, (0, 0, 255), -1)

                    cv2.imshow("Kamera Görüntüsü", image_bgr)

                except Exception:
                    traceback.print_exc()

        finally:
            print("Kapanıyor...")
            vis.destroy_window()
            cv2.destroyAllWindows()
            self.zed.close()


if __name__ == "__main__":
    # ilk deneme için voxel 20 mm; gerekirse düşür
    detector = StoneDimensionEstimator(
        voxel_size_mm=20.0,
        target_dims_cm=(30.0, 20.0, 10.0),
        allow_rotations=True
    )
    detector.run_customer_demo()
