import sys
import pyzed.sl as sl
import numpy as np
import open3d as o3d
import cv2
import traceback

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
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            point_cloud_np = self.point_cloud.get_data()
            points = point_cloud_np[:, :, :3]
            mask = np.isfinite(points).all(axis=2)
            valid_points = points[mask]
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            image_np = self.image.get_data()
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            return valid_points, image_bgr
        return None, None

    def process_point_cloud(self, points_np, voxel_size=5.0):
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
        labels = np.array(objects_pcd.cluster_dbscan(eps=25.0, min_points=25, print_progress=False))
        if labels.max() == -1:
            return pcd_clean, None
        counts = np.bincount(labels[labels >= 0])
        if len(counts) == 0:
            return pcd_clean, None
        largest_cluster_label = np.argmax(counts)
        stone_indices = np.where(labels == largest_cluster_label)[0]
        stone_pcd = objects_pcd.select_by_index(stone_indices)
        return pcd_clean, stone_pcd

    def calculate_measurements(self, stone_pcd):
        if stone_pcd is None or len(stone_pcd.points) < 10:
            return None
        obb = stone_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)
        dimensions = obb.extent
        volume = 0
        try:
            hull, _ = stone_pcd.compute_convex_hull()
            if hull.is_watertight():
                volume = hull.get_volume()
        except:
            pass
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
        if stone_pcd is None or len(stone_pcd.points) < 20:
            return None
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(stone_pcd, alpha)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
            return mesh
        except:
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
        vis = o3d.visualization.Visualizer()
        vis.create_window("3D Analiz", width=640, height=480, left=650, top=50)
        vis.get_render_option().background_color = np.asarray([0.15, 0.15, 0.18])
        vis.get_render_option().mesh_show_back_face = True
        view_control = vis.get_view_control()

        try:
            while True:
                try:
                    points_np, image_bgr = self.get_point_cloud_and_image()
                    if points_np is None or len(points_np) < 50:
                        continue

                    _, stone_pcd = self.process_point_cloud(points_np)
                    measurements = self.calculate_measurements(stone_pcd)
                    stone_mesh = self.create_stone_mesh(stone_pcd)

                    vis.clear_geometries()

                    if stone_mesh:
                        vis.add_geometry(stone_mesh)

                    if measurements:
                        obb = measurements["oriented_bounding_box"]
                        vis.add_geometry(obb)

                        dims = measurements["dimensions_mm"]
                        dim_text = f"{dims[0]/10:.1f} x {dims[1]/10:.1f} x {dims[2]/10:.1f} cm"
                        volume_text = f"Hacim: {measurements['volume_cm3']:.0f} cm³"
                        weight_text = f"Ağırlık: {measurements['weight_kg']:.2f} kg"

                        # Metinleri göster
                        cv2.putText(image_bgr, dim_text, (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(image_bgr, volume_text, (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(image_bgr, weight_text, (30, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        # Ölçülen taşın ortasını işaretle (kırmızı nokta)
                        center_3d = obb.get_center()
                        center_2d = self.project_3d_to_2d(center_3d)
                        if center_2d:
                            cv2.circle(image_bgr, center_2d, 8, (0, 0, 255), -1)

                    view_control.rotate(2.0, 0.0)
                    vis.poll_events()
                    vis.update_renderer()

                    cv2.imshow("Kamera Görüntüsü", image_bgr)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    traceback.print_exc()

        finally:
            vis.destroy_window()
            cv2.destroyAllWindows()
            self.zed.close()


if __name__ == "__main__":
    detector = StoneDimensionEstimator()
    detector.run_customer_demo()
