# zed_packing_with_calib_OBB_aligned_support_aware.py
# --- Inside-envelope version (bugfix) ---
# Bu sürümde "zarf" (envelope) taşın İÇİNE oturtulur ve dışarı taşmaz.
# Güncellemeler / Düzeltmeler:
#  - İç zarf, voxel grid üzerinde morfolojik erosion ile çıkarılır (offset_voxels_inside kadar içe).
#  - Paketleme bu iç zarf üzerinde yapılır.
#  - Aç/Kapa kısayolları (B=Zarf, P=Kutu, M=Taş, N=Kolon) eklendi.
#  - Open3D mesh kopyalama/transform güvenli hale getirildi (clone/copy yoksa manuel kopya).
#  - Erosion için SciPy varsa generate_binary_structure kullanılır; yoksa sağlam NumPy fallback.
#  - Küçük hatalar ve değişken tutarsızlıkları giderildi.

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
                 max_voxels=60_000_000,
                 offset_voxels_inside=2):
        self.voxel_size_mm = float(voxel_size_mm)
        self.target_dims_cm = tuple(target_dims_cm)  # (L,W,H) in cm
        self.allow_rotations = bool(allow_rotations)
        self.max_voxels = int(max_voxels)
        # İç zarf için yüzeyden kaç voxel içeri gidileceği
        self.offset_voxels_inside = int(max(0, offset_voxels_inside))

        self.scale_factor = 1.0  # kalibrasyonla değişir

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
        print(f"Voxel: {self.voxel_size_mm} mm | Target (cm): {self.target_dims_cm} | Rot: {self.allow_rotations} | Inside offset (vox): {self.offset_voxels_inside}")

        # Kalibrasyon yardımcıları
        self.last_xyz_map = None
        self.last_bgr = None
        self.calib_clicks = []

        # Durum
        self.captured_data = None
        self.analysis_results = None

        # Görünürlük bayrakları
        self.flags = {
            'envelope': True,    # İç OBB zarfı
            'pieces': True,      # Yerleştirilen kutular
            'stone': True,       # Taş mesh
            'columns': True      # Kırmızı yüzey kolonları
        }

    # ---------- ZED helpers ----------
    def get_point_cloud_and_image(self):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            pc_raw = self.point_cloud.get_data().astype(np.float32)  # HxWx4
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

    # ---------- Point cloud processing ----------
    def process_point_cloud(self, points_np, voxel_size=5.0):
        if points_np is None or len(points_np) == 0:
            return None, None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        if len(pcd_down.points) == 0:
            return pcd_down, None
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # Zemin ayır
        try:
            _, inliers = pcd_clean.segment_plane(distance_threshold=15.0, ransac_n=3, num_iterations=1000)
            objects_pcd = pcd_clean.select_by_index(inliers, invert=True)
        except Exception:
            objects_pcd = pcd_clean
        if len(objects_pcd.points) == 0:
            return pcd_clean, None
        # En büyük küme
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

    # ---------- Utils ----------
    @staticmethod
    def _mesh_copy(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Open3D'de güvenli kopya: clone/copy yoksa manuel kopyalar."""
        try:
            return mesh.clone()
        except Exception:
            try:
                return mesh.copy()
            except Exception:
                m2 = o3d.geometry.TriangleMesh()
                m2.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
                m2.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
                if mesh.has_vertex_normals():
                    m2.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
                if mesh.has_vertex_colors():
                    m2.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
                return m2

    @staticmethod
    def _mesh_transform(mesh, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        mesh2 = StoneDimensionEstimator._mesh_copy(mesh)
        mesh2.transform(T)
        return mesh2

    # ---------- Voxelization (full "inside" grid) ----------
    def voxelize_mesh_to_bool(self, mesh, voxel_size_mm):
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
                pass

        return grid, minb, (nx, ny, nz)

    # ---------- 3D Binary Erosion ----------
    def erode_grid(self, grid, iterations=1, connectivity=6):
        """3D binary erosion. SciPy varsa onu, yoksa NumPy fallback kullanır."""
        try:
            from scipy.ndimage import binary_erosion, generate_binary_structure
            if connectivity in (6, 18, 26):
                conn = {6: 1, 18: 2, 26: 3}[connectivity]
                structure = generate_binary_structure(3, conn)
            else:
                structure = generate_binary_structure(3, 1)
            return binary_erosion(grid.astype(bool), structure=structure, iterations=int(max(1, iterations)), border_value=0)
        except Exception:
            # NumPy fallback: 6-bağlı tek iterasyonlu erosion benzeri
            def erode_once(m):
                # Bir vokselin kalması için tüm 6 komşusu ve kendisi True olmalı
                m6 = m.copy()
                for ax in (0, 1, 2):
                    m6 &= np.roll(m, 1, axis=ax)
                    m6 &= np.roll(m, -1, axis=ax)
                # Kenar taşmalarını sil
                m6[0, :, :] = False; m6[-1, :, :] = False
                m6[:, 0, :] = False; m6[:, -1, :] = False
                m6[:, :, 0] = False; m6[:, :, -1] = False
                return m6
            out = grid.astype(bool)
            for _ in range(int(max(1, iterations))):
                out = erode_once(out)
            return out

    # ---------- Surface-aware support (INSIDE shift) ----------
    def build_surface_support(self, grid, offset_voxels_inside=0):
        """
        grid: (nz,ny,nx) True=taşın içinde (OBB-lokal)
        Dönüş:
          top_idx: (ny,nx) her kolon için en üstteki True voxel'ın z indexi,
                   ancak 'offset_voxels_inside' kadar AŞAĞI (taşın içine) kaydırılmış.
        """
        nz, ny, nx = grid.shape
        top_idx = np.full((ny, nx), -1, dtype=int)

        for iy in range(ny):
            col = grid[:, iy, :]
            rev = col[::-1, :]
            first_true = np.argmax(rev, axis=0)
            has_true = np.any(rev, axis=0)
            ti = np.where(has_true, (nz - 1 - first_true), -1)
            if offset_voxels_inside > 0:
                ti = np.where(ti >= 0, np.maximum(ti - offset_voxels_inside, 0), -1)
            top_idx[iy, :] = ti
        return top_idx

    # ---------- Support-aware packing ----------
    def greedy_pack_support_aware(self, grid, piece_vox_shape, top_idx):
        nz, ny, nx = grid.shape
        pz, py, px = piece_vox_shape

        filled = np.zeros_like(grid, dtype=bool)
        placed = []

        rots = [(pz, py, px)]
        if self.allow_rotations:
            from itertools import permutations
            rots = list({tuple(x) for x in permutations((pz, py, px))})

        for (pz_, py_, px_) in rots:
            for iy in range(ny - py_ + 1):
                for ix in range(nx - px_ + 1):
                    local_top = top_idx[iy:iy + py_, ix:ix + px_]
                    if np.any(local_top < 0):
                        continue
                    z_top = int(np.min(local_top))
                    z_bot = z_top - pz_ + 1
                    if z_bot < 0:
                        continue
                    sub_grid = grid[z_bot:z_top + 1, iy:iy + py_, ix:ix + px_]
                    if sub_grid.shape != (pz_, py_, px_):
                        continue
                    region_ok = bool(np.all(sub_grid))
                    region_free = not bool(np.any(filled[z_bot:z_top + 1, iy:iy + py_, ix:ix + px_]))
                    if region_ok and region_free:
                        filled[z_bot:z_top + 1, iy:iy + py_, ix:ix + px_] = True
                        placed.append((ix, iy, z_bot, px_, py_, pz_))
        used_vox = int(np.count_nonzero(filled))
        return placed, used_vox, filled

    # ---------- Calibration ----------
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
                    cv2.setMouseCallback("Kamera Görüntüsü", lambda *args: None)

    def _pixel_to_3d(self, xy, k=2):
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
        return np.median(patch[valid], axis=0)

    def _project_3d_to_pixel(self, p3):
        fx, fy, cx, cy = self.camera_intrinsics.fx, self.camera_intrinsics.fy, self.camera_intrinsics.cx, self.camera_intrinsics.cy
        x, y, z = float(p3[0]), float(p3[1]), float(p3[2])
        if z <= 0 or not np.isfinite(z):
            return None
        u = int((x * fx / z) + cx); v = int((y * fy / z) + cy)
        return (u, v)

    # ---------- Full analysis ----------
    def _perform_full_analysis(self, points_np):
        print("1/6 - Nokta bulutu işleniyor...")
        _, stone_pcd = self.process_point_cloud(points_np)

        print("2/6 - Ölçümler hesaplanıyor...")
        measurements = self.calculate_measurements(stone_pcd)

        print("3/6 - Mesh oluşturuluyor...")
        stone_mesh = self.create_stone_mesh(stone_pcd)

        results = {
            "stone_pcd": stone_pcd,
            "measurements": measurements,
            "stone_mesh": stone_mesh,
            "placed_boxes_world": [],
            "placed_count": 0,
            "theoretical_by_volume": 0,
            "efficiency": 0.0,
            "total_vol_cm3": 0.0,
            "red_prism_meshes": [],
            "inner_envelope_mesh": None
        }

        if stone_mesh is not None and measurements is not None:
            try:
                # OBB'ye hizala (lokal eksenler)
                obb = measurements["oriented_bounding_box"]
                R = obb.R
                c = np.asarray(obb.center)
                stone_local = self._mesh_transform(stone_mesh, R.T, -R.T @ c)

                print("4/6 - Voxelization...")
                grid, origin_local, (nx, ny, nz) = self.voxelize_mesh_to_bool(stone_local, self.voxel_size_mm)

                total_inside = int(np.count_nonzero(grid))
                total_vol_mm3 = total_inside * (self.voxel_size_mm ** 3)
                results["total_vol_cm3"] = total_vol_mm3 / 1000.0

                # ---- İç zarf için erosion ----
                if self.offset_voxels_inside > 0:
                    inner_grid = self.erode_grid(grid, iterations=self.offset_voxels_inside, connectivity=6)
                else:
                    inner_grid = grid.copy()

                # İç zarf için top index (kırmızı sütunlar)
                top_idx = self.build_surface_support(inner_grid, offset_voxels_inside=0)  # erosion sonrası

                # Kırmızı yüzey prizması (1-voxel sütunlar) - İÇ gridden
                red_meshes = []
                aabb_local = stone_local.get_axis_aligned_bounding_box()
                minb = np.asarray(aabb_local.min_bound)
                for iy in range(ny):
                    for ix in range(nx):
                        tz = top_idx[iy, ix]
                        if tz < 0:
                            continue
                        vx = minb[0] + (ix + 0.5) * self.voxel_size_mm
                        vy = minb[1] + (iy + 0.5) * self.voxel_size_mm
                        vz = minb[2] + (tz + 0.5) * self.voxel_size_mm
                        box = o3d.geometry.TriangleMesh.create_box(
                            width=self.voxel_size_mm, height=self.voxel_size_mm, depth=self.voxel_size_mm
                        )
                        box.translate((vx - self.voxel_size_mm/2,
                                       vy - self.voxel_size_mm/2,
                                       vz - self.voxel_size_mm/2))
                        box.paint_uniform_color([0.85, 0.1, 0.1])
                        box_w = self._mesh_transform(box, R, c)
                        red_meshes.append(box_w)
                results["red_prism_meshes"] = red_meshes

                # ---- İÇ OBB (zarf) hesapla ----
                pz_idx, py_idx, px_idx = np.where(inner_grid)  # (z,y,x)
                if pz_idx.size > 0:
                    vx = minb[0] + (px_idx + 0.5) * self.voxel_size_mm
                    vy = minb[1] + (py_idx + 0.5) * self.voxel_size_mm
                    vz = minb[2] + (pz_idx + 0.5) * self.voxel_size_mm
                    pts_local = np.column_stack((vx, vy, vz))

                    pcd_inner = o3d.geometry.PointCloud()
                    pcd_inner.points = o3d.utility.Vector3dVector(pts_local)
                    inner_obb_local = pcd_inner.get_oriented_bounding_box()

                    # Inner OBB'yi dünyaya taşı ve kırmızı kutu olarak çiz
                    inner_box = o3d.geometry.TriangleMesh.create_box(*inner_obb_local.extent)
                    inner_box.paint_uniform_color([0.9, 0.1, 0.1])
                    inner_box.translate(-inner_obb_local.extent / 2.0)
                    T_local = np.eye(4)
                    T_local[:3, :3] = inner_obb_local.R
                    T_local[:3, 3] = inner_obb_local.center
                    inner_box.transform(T_local)
                    inner_box = self._mesh_transform(inner_box, R, c)
                    results["inner_envelope_mesh"] = inner_box

                # ---- Paketleme: İÇ grid üzerinde ----
                tx_mm, ty_mm, tz_mm = [d * 10.0 for d in self.target_dims_cm]
                px = max(1, int(np.ceil(tx_mm / self.voxel_size_mm)))
                py = max(1, int(np.ceil(ty_mm / self.voxel_size_mm)))
                pz = max(1, int(np.ceil(tz_mm / self.voxel_size_mm)))

                print("5/6 - Destek-bilinçli kutu yerleştirme (İÇ zarf)...")
                placed, used_vox, _ = self.greedy_pack_support_aware(inner_grid, (pz, py, px), top_idx)

                results["placed_count"] = len(placed)
                results["theoretical_by_volume"] = int(
                    math.floor(total_vol_mm3 / max((px * py * pz) * (self.voxel_size_mm ** 3), 1e-6))
                )
                inner_inside = max(int(np.count_nonzero(inner_grid)), 1)
                results["efficiency"] = (used_vox / inner_inside) * 100.0

                # Kutuları dünya koordinatlarına taşı ve görselleştir
                placed_boxes = []
                for (ix, iy, iz_bot, sx, sy, sz) in placed:
                    lx = origin_local[0] + ix * self.voxel_size_mm
                    ly = origin_local[1] + iy * self.voxel_size_mm
                    lz = origin_local[2] + iz_bot * self.voxel_size_mm
                    wx = sx * self.voxel_size_mm
                    wy = sy * self.voxel_size_mm
                    wz = sz * self.voxel_size_mm

                    box_local = o3d.geometry.TriangleMesh.create_box(width=wx, height=wy, depth=wz)
                    box_local.compute_vertex_normals()
                    box_local.paint_uniform_color([0.1, 0.9, 0.1])
                    box_local.translate((lx, ly, lz))
                    box_world = self._mesh_transform(box_local, R, c)

                    min_world = (R @ np.array([lx, ly, lz])) + c
                    center_local = np.array([lx + wx / 2.0, ly + wy / 2.0, lz + wz / 2.0])
                    center_world = (R @ center_local) + c

                    placed_boxes.append({
                        "mesh": box_world,
                        "csv_data": (min_world[0], min_world[1], min_world[2], wx, wy, wz, center_world)
                    })
                results["placed_boxes_world"] = placed_boxes

            except MemoryError as me:
                print(f"HATA: Voxelization için bellek yetersiz: {me}")
            except Exception:
                traceback.print_exc()

        print("6/6 - Analiz tamamlandı.")
        return results

    # ---------- Main loop ----------
    def _refresh_scene(self, vis):
        """Görünürlük bayraklarına göre sahneyi yeniden kur."""
        vis.clear_geometries()
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=100))
        if self.analysis_results is None:
            vis.poll_events(); vis.update_renderer()
            return
        res = self.analysis_results
        if self.flags['stone'] and res['stone_mesh']:
            vis.add_geometry(res['stone_mesh'])
        if self.flags['envelope'] and res['inner_envelope_mesh'] is not None:
            vis.add_geometry(res['inner_envelope_mesh'])
        if self.flags['columns']:
            for rm in res['red_prism_meshes']:
                vis.add_geometry(rm)
        if self.flags['pieces']:
            for box_data in res['placed_boxes_world']:
                vis.add_geometry(box_data['mesh'])
        vis.poll_events(); vis.update_renderer()

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
        print(" B: İç OBB zarfını aç/kapat")
        print(" P: Yerleştirilen kutuları aç/kapat")
        print(" M: Taş mesh aç/kapat")
        print(" N: Kırmızı yüzey sütunlarını aç/kapat")
        print(" Q: Çıkış")
        print("----------------\n")

        should_exit = {"v": False}
        def cb_q(_): should_exit["v"] = True; return False
        vis.register_key_callback(ord("Q"), cb_q)

        try:
            while not should_exit["v"]:
                # CANLI
                if self.captured_data is None:
                    points_np, image_bgr = self.get_point_cloud_and_image()
                    if image_bgr is None:
                        continue
                    h, w, _ = image_bgr.shape
                    cv2.putText(image_bgr, "CANLI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image_bgr, "Analiz icin 'C' tusuna basin", (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    for (cx, cy) in self.calib_clicks:
                        cv2.circle(image_bgr, (cx, cy), 6, (0, 0, 255), -1)

                    cv2.imshow("Kamera Görüntüsü", image_bgr)

                    self._refresh_scene(vis)

                # ANALIZ
                else:
                    if self.analysis_results is None:
                        img_wait = self.captured_data['image'].copy()
                        h, w, _ = img_wait.shape
                        cv2.putText(img_wait, "ANALIZ EDILIYOR...", (w // 2 - 200, h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.imshow("Kamera Görüntüsü", img_wait)
                        cv2.waitKey(1)

                        self.analysis_results = self._perform_full_analysis(self.captured_data['points'])
                        self._refresh_scene(vis)

                    # Overlay
                    image_display = self.captured_data['image'].copy()
                    h, w = image_display.shape[:2]
                    panel_h = 190
                    cv2.rectangle(image_display, (0, 0), (w, panel_h), (0, 0, 0), -1)
                    y = 25
                    res = self.analysis_results
                    if res['measurements']:
                        dims = res['measurements']["dimensions_mm"]
                        cv2.putText(image_display, f"Tas OBB (cm): {dims[0]/10:.1f}x{dims[1]/10:.1f}x{dims[2]/10:.1f}",
                                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 22
                    cv2.putText(image_display, f"Hedef (cm): {self.target_dims_cm[0]:.1f}x{self.target_dims_cm[1]:.1f}x{self.target_dims_cm[2]:.1f}",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1); y += 20
                    if res['total_vol_cm3'] > 0:
                        cv2.putText(image_display, f"Hacim~ (cm3): {res['total_vol_cm3']:.0f}",
                                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); y += 22
                    cv2.putText(image_display, f"Yerlesen: {res['placed_count']} | Teorik: {res['theoretical_by_volume']} | Verim: {res['efficiency']:.1f}%",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2); y += 22
                    cv2.putText(image_display, f"Mod: İc zarf (erosion={self.offset_voxels_inside} vox)",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 1); y += 20
                    cv2.putText(image_display, "B: Zarf  P: Kutu  M: Tas  N: Kolon  |  R: Reset  E: CSV  F: Kalibrasyon",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                    # Kutu merkezlerini 2D'ye yaz
                    for idx, box_data in enumerate(res['placed_boxes_world'], start=1):
                        pt = self._project_3d_to_pixel(box_data['csv_data'][6])  # center_world
                        if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                            cv2.circle(image_display, pt, 4, (0, 0, 255), -1)
                            cv2.putText(image_display, f"{idx}", (pt[0]+6, pt[1]-6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    cv2.imshow("Kamera Görüntüsü", image_display)

                    vis.poll_events(); vis.update_renderer()

                # Tuşlar
                k = cv2.waitKey(10) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('c') and self.captured_data is None:
                    print("\nGoruntu yakalandi, analiz baslatiliyor...")
                    points_np, image_bgr = self.get_point_cloud_and_image()
                    self.captured_data = {'points': points_np, 'image': image_bgr}
                elif k == ord('r'):
                    print("\nAnaliz sifirlandi, canli goruntuye donuluyor.")
                    self.captured_data = None
                    self.analysis_results = None
                    self._refresh_scene(vis)
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
                                    vx, vy, vz, sx, sy, sz, cc = box['csv_data']
                                    f.write(f"{vx:.2f},{vy:.2f},{vz:.2f},{sx:.2f},{sy:.2f},{sz:.2f},{cc[0]:.2f},{cc[1]:.2f},{cc[2]:.2f}\n")
                            print(f"Kesim plani CSV kaydedildi: {fname}")
                        except Exception:
                            traceback.print_exc()
                    else:
                        print("CSV'ye aktarilacak analiz sonucu bulunamadi.")
                elif k == ord('b'):
                    self.flags['envelope'] = not self.flags['envelope']
                    print(f"Envelope {'ON' if self.flags['envelope'] else 'OFF'}")
                    self._refresh_scene(vis)
                elif k == ord('p'):
                    self.flags['pieces'] = not self.flags['pieces']
                    print(f"Pieces {'ON' if self.flags['pieces'] else 'OFF'}")
                    self._refresh_scene(vis)
                elif k == ord('m'):
                    self.flags['stone'] = not self.flags['stone']
                    print(f"Stone {'ON' if self.flags['stone'] else 'OFF'}")
                    self._refresh_scene(vis)
                elif k == ord('n'):
                    self.flags['columns'] = not self.flags['columns']
                    print(f"Columns {'ON' if self.flags['columns'] else 'OFF'}")
                    self._refresh_scene(vis)

        finally:
            print("Kapanıyor...")
            vis.destroy_window()
            cv2.destroyAllWindows()
            self.zed.close()


if __name__ == "__main__":
    # Örnek: 30x20x10 cm hedef kutu
    detector = StoneDimensionEstimator(
        voxel_size_mm=20.0,
        target_dims_cm=(10.0, 10.0, 5.0),
        allow_rotations=True,
        offset_voxels_inside=2  # iç zarf için erosion miktarı (voxel)
    )
    detector.run()
