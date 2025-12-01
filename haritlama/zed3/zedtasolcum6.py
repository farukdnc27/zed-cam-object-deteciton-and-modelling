# zed_packing.py
import sys
import os
import datetime
import traceback
import math

import numpy as np
import cv2
import open3d as o3d
import pyzed.sl as sl

# Optional robust geometry lib
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
        target_dims_cm: hedef parça (cm) tuple (L, W, H) - senin kullanımında cm giriyorsun
        allow_rotations: hedef parça için 90° rotasyonlara izin verilsin mi
        max_voxels: belleği korumak için izin verilen maksimum voxel sayısı (nz*ny*nx)
        """
        self.voxel_size_mm = float(voxel_size_mm)
        self.target_dims_cm = tuple(target_dims_cm)
        self.allow_rotations = bool(allow_rotations)
        self.max_voxels = int(max_voxels)

        # ZED init (kendi ayarların)
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
        print(f"Voxel size: {self.voxel_size_mm} mm, target dims (cm): {self.target_dims_cm}, rotations: {self.allow_rotations}")

    # ---------- ZED helpers ----------
    def get_point_cloud_and_image(self):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            pc = self.point_cloud.get_data()
            points = pc[:, :, :3].astype(np.float32)
            mask = np.isfinite(points).all(axis=2)
            valid_points = points[mask]
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            img = self.image.get_data()
            try:
                image_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            except Exception:
                image_bgr = img[..., :3][:, :, ::-1]
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
            plane_model, inliers = pcd_clean.segment_plane(distance_threshold=15.0, ransac_n=3, num_iterations=1000)
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
        largest_cluster_label = np.argmax(counts)
        stone_indices = np.where(labels == largest_cluster_label)[0]
        stone_pcd = objects_pcd.select_by_index(stone_indices)
        return pcd_clean, stone_pcd

    def calculate_measurements(self, stone_pcd):
        if stone_pcd is None or len(stone_pcd.points) < 10:
            return None
        obb = stone_pcd.get_oriented_bounding_box()
        obb.color = (1.0, 0.0, 0.0)
        dimensions = np.asarray(obb.extent)  # mm
        # try alpha-shape volume via create_stone_mesh then compute approx volume
        volume_mm3 = 0.0
        try:
            mesh = self.create_stone_mesh(stone_pcd, alpha=25.0)
            if mesh is not None:
                # sample points inside by voxelization later; for summary we may try hull
                hull, _ = stone_pcd.compute_convex_hull()
                if hasattr(hull, "get_volume"):
                    try:
                        volume_mm3 = hull.get_volume()
                    except Exception:
                        volume_mm3 = 0.0
        except Exception:
            volume_mm3 = 0.0

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

    # ---------- Voxelization and packing ----------
    def voxelize_mesh_to_bool(self, mesh, voxel_size_mm):
        """
        Return (grid_bool, origin, nx, ny, nz)
        grid shape: (nz, ny, nx) boolean where True means inside mesh (available volume).
        origin is min_bound in mesh coordinates for (x,y,z) of grid[0,0,0] voxel corner.
        """
        # axis aligned bounding box (in mm)
        aabb = mesh.get_axis_aligned_bounding_box()
        minb = np.asarray(aabb.min_bound)
        maxb = np.asarray(aabb.max_bound)
        dims = (maxb - minb)
        nx = int(np.ceil(dims[0] / voxel_size_mm))
        ny = int(np.ceil(dims[1] / voxel_size_mm))
        nz = int(np.ceil(dims[2] / voxel_size_mm))
        total_vox = int(nx) * int(ny) * int(nz)
        if total_vox > self.max_voxels:
            raise MemoryError(f"Voxel grid çok büyük: {total_vox} voxels. voxel_size_mm arttır veya max_voxels arttır.")

        grid = np.zeros((nz, ny, nx), dtype=bool)

        # Build center coordinates for each voxel slice
        xs = (np.arange(nx) + 0.5) * voxel_size_mm + minb[0]
        ys = (np.arange(ny) + 0.5) * voxel_size_mm + minb[1]
        zs = (np.arange(nz) + 0.5) * voxel_size_mm + minb[2]

        # Use trimesh.contains if available (more robust)
        if _HAS_TRIMESH:
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            # trimesh expects float units; keep mm as-is
            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            # do slice by slice to avoid huge memory
            for iz, z in enumerate(zs):
                yy, xx = np.meshgrid(ys, xs, indexing='ij')
                pts = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
                try:
                    inside = tm.contains(pts)
                except Exception:
                    # fallback: mark none
                    inside = np.zeros(pts.shape[0], dtype=bool)
                grid[iz, :, :] = inside.reshape((ny, nx))
            return grid, minb, (nx, ny, nz)
        else:
            # Fallback approximate method: sample many surface points and KDTree
            sampled = mesh.sample_points_uniformly(number_of_points=200000)
            sampled_pts = np.asarray(sampled.points)
            if sampled_pts.shape[0] == 0:
                return grid, minb, (nx, ny, nz)
            try:
                from sklearn.neighbors import KDTree
                tree = KDTree(sampled_pts)
            except Exception:
                # If sklearn missing, fallback to naive nearest search (slow)
                tree = None
            diag = np.sqrt(3 * (voxel_size_mm ** 2)) / 2.0
            for iz, z in enumerate(zs):
                yy, xx = np.meshgrid(ys, xs, indexing='ij')
                centers = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
                if tree is not None:
                    dists, _ = tree.query(centers, k=1)
                    inside = (dists.ravel() < diag * 1.15)
                else:
                    # slow loop
                    inside = []
                    for c in centers:
                        dmin = np.min(np.linalg.norm(sampled_pts - c, axis=1))
                        inside.append(dmin < diag * 1.15)
                    inside = np.array(inside, dtype=bool)
                grid[iz, :, :] = inside.reshape((ny, nx))
            return grid, minb, (nx, ny, nz)

    def greedy_pack(self, grid, piece_vox_shape):
        """
        grid: boolean (nz,ny,nx) True=free/inside
        piece_vox_shape: (pz,py,px) in voxels
        returns: placed list (voxel coords) and filled grid
        """
        nz, ny, nx = grid.shape
        pz0, py0, px0 = piece_vox_shape
        # generate rotation variants (permutations) if allowed
        rotations = []
        if self.allow_rotations:
            from itertools import permutations
            rotations = list({tuple(x) for x in permutations((pz0, py0, px0))})
        else:
            rotations = [(pz0, py0, px0)]
        filled = np.zeros_like(grid, dtype=bool)
        placed = []
        # greedy scan order z,y,x
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    if not grid[iz, iy, ix] or filled[iz, iy, ix]:
                        continue
                    placed_flag = False
                    for (pz, py, px) in rotations:
                        iz2 = iz + pz
                        iy2 = iy + py
                        ix2 = ix + px
                        if iz2 > nz or iy2 > ny or ix2 > nx:
                            continue
                        region = grid[iz:iz2, iy:iy2, ix:ix2]
                        if region.shape != (pz, py, px):
                            continue
                        if np.all(region) and not np.any(filled[iz:iz2, iy:iy2, ix:ix2]):
                            filled[iz:iz2, iy:iy2, ix:ix2] = True
                            placed.append((ix, iy, iz, px, py, pz))  # store in voxel coords (x,y,z,size)
                            placed_flag = True
                            break
                    if placed_flag:
                        continue
        used_vox = int(np.count_nonzero(filled))
        return placed, used_vox, filled

    # ---------- run & integrate ----------
    def run_customer_demo(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("3D Analiz", width=1100, height=700, left=50, top=50)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.18])
        opt.mesh_show_back_face = True
        view_control = vis.get_view_control()

        print("Kısayollar: R=realtime, S=snapshot, Q=quit")
        should_exit = {"val": False}
        def cb_q(vis_obj):
            should_exit["val"] = True
            return False
        vis.register_key_callback(ord("Q"), cb_q)

        last_mesh = {"mesh": None}
        last_image = {"img": None}
        snapshot_prepared = {"val": False}
        mode = {"value": "realtime"}

        try:
            while not should_exit["val"]:
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break

                if mode["value"] == "realtime":
                    try:
                        points_np, image_bgr = self.get_point_cloud_and_image()
                        if points_np is None or len(points_np) < 200:
                            # gösterim için boş ekran
                            cv2.imshow("Kamera Görüntüsü", np.zeros((480, 640, 3), dtype=np.uint8))
                            vis.poll_events(); vis.update_renderer()
                            continue

                        _, stone_pcd = self.process_point_cloud(points_np)
                        measurements = self.calculate_measurements(stone_pcd)
                        stone_mesh = self.create_stone_mesh(stone_pcd)

                        last_mesh["mesh"] = stone_mesh
                        last_image["img"] = image_bgr.copy()

                        # --- packing step (voxelize + greedy) ---
                        placed_boxes = []
                        used_vox = 0
                        total_inside_vox = 0
                        if stone_mesh is not None:
                            try:
                                grid, origin, (nx, ny, nz) = self.voxelize_mesh_to_bool(stone_mesh, self.voxel_size_mm)
                                # grid is (nz,ny,nx)
                                total_inside_vox = int(np.count_nonzero(grid))
                                # target piece in mm:
                                tx_mm = self.target_dims_cm[0] * 10.0
                                ty_mm = self.target_dims_cm[1] * 10.0
                                tz_mm = self.target_dims_cm[2] * 10.0
                                # convert to voxels (round up)
                                px = max(1, int(np.ceil(tx_mm / self.voxel_size_mm)))
                                py = max(1, int(np.ceil(ty_mm / self.voxel_size_mm)))
                                pz = max(1, int(np.ceil(tz_mm / self.voxel_size_mm)))
                                # pack
                                placed_boxes, used_vox, filled = self.greedy_pack(grid, (pz, py, px))
                            except MemoryError as me:
                                print("Hata (voxel grid büyük):", me)
                            except Exception:
                                traceback.print_exc()

                        # --- visualize results ---
                        vis.clear_geometries()
                        if stone_mesh is not None:
                            vis.add_geometry(stone_mesh)
                        if measurements is not None:
                            obb = measurements["oriented_bounding_box"]
                            vis.add_geometry(obb)

                        # add placed boxes as Open3D boxes (create_box uses width,height,depth)
                        if len(placed_boxes) > 0:
                            # origin = min bound, vox centers computed earlier: place at voxel cell centers
                            aabb = stone_mesh.get_axis_aligned_bounding_box()
                            minb = np.asarray(aabb.min_bound)
                            # note grid shape (nz,ny,nx) -> placed boxes stored as (ix,iy,iz, px,py,pz) where ix is x index
                            for (ix, iy, iz, sx, sy, sz) in placed_boxes:
                                # compute voxel corner (min) in mm
                                vx = minb[0] + ix * self.voxel_size_mm
                                vy = minb[1] + iy * self.voxel_size_mm
                                vz = minb[2] + iz * self.voxel_size_mm
                                # box physical size:
                                wx = sx * self.voxel_size_mm
                                wy = sy * self.voxel_size_mm
                                wz = sz * self.voxel_size_mm
                                # create box and translate to corner
                                box = o3d.geometry.TriangleMesh.create_box(width=wx, height=wy, depth=wz)
                                box.compute_vertex_normals()
                                # color them semi-distinct
                                box.paint_uniform_color([0.8, 0.3, 0.3])
                                # Open3D box origin at (0,0,0) -> translate to voxel corner
                                box.translate((vx, vy, vz))
                                vis.add_geometry(box)

                        # keep camera pose if possible
                        try:
                            prev = view_control.convert_to_pinhole_camera_parameters()
                        except Exception:
                            prev = None

                        if prev is not None:
                            try:
                                view_control.convert_from_pinhole_camera_parameters(prev)
                            except Exception:
                                pass

                        vis.poll_events(); vis.update_renderer()

                        # --- 2D overlay info ---
                        if measurements is not None:
                            dims = measurements["dimensions_mm"]
                            dim_text = f"Taş OBB (cm): {dims[0]/10:.1f} x {dims[1]/10:.1f} x {dims[2]/10:.1f}"
                            cv2.putText(image_bgr, dim_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        if total_inside_vox > 0:
                            total_vol_mm3 = total_inside_vox * (self.voxel_size_mm ** 3)
                            total_vol_cm3 = total_vol_mm3 / 1000.0
                            piece_vol_mm3 = (px * py * pz) * (self.voxel_size_mm ** 3)
                            piece_vol_cm3 = piece_vol_mm3 / 1000.0
                            theoretical_by_volume = int(math.floor((total_vol_mm3) / piece_vol_mm3)) if piece_vol_mm3>0 else 0
                            placed_count = len(placed_boxes)
                            used_vol_mm3 = used_vox * (self.voxel_size_mm ** 3)
                            used_vol_cm3 = used_vol_mm3 / 1000.0
                            efficiency = (used_vol_mm3 / total_vol_mm3 * 100.0) if total_vol_mm3>0 else 0.0
                            cv2.putText(image_bgr, f"Hacim (cm3): {total_vol_cm3:.0f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                            cv2.putText(image_bgr, f"Hedef parca adedi (greedy): {placed_count}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                            cv2.putText(image_bgr, f"Teorik (hacme gore): {theoretical_by_volume}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                            cv2.putText(image_bgr, f"Verim: {efficiency:.1f}%", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,200), 2)
                        cv2.imshow("Kamera Görüntüsü", image_bgr)

                    except Exception:
                        traceback.print_exc()
                        # hata olsa da döngü devam eder
                else:
                    # snapshot mode - not implemented more here
                    vis.poll_events(); vis.update_renderer()
                    cv2.imshow("Kamera Görüntüsü", np.zeros((480,640,3), dtype=np.uint8))

        finally:
            print("Kapanıyor...")
            vis.destroy_window()
            cv2.destroyAllWindows()
            self.zed.close()


if __name__ == "__main__":
    # örnek: voxel 20mm, hedef ölçü 30x20x10 cm
    detector = StoneDimensionEstimator(voxel_size_mm=20.0, target_dims_cm=(30.0, 20.0, 10.0), allow_rotations=True)
    detector.run_customer_demo()
