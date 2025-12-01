import sys
import time
import pyzed.sl as sl
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import open3d as o3d
import os
import traceback


class SimpleViewer:
    def __init__(self):
        self.is_running = True
        self.chunks_updated_flag = True
        self.toggle_mapping_flag = False
        self.mesh_data = None
        self.show_3d_window = False
        self.point_cloud_o3d = None
        self.mapping_completed = False

    def init(self, calibration_params, mesh, is_mesh):
        self.is_mesh = is_mesh
        print("Simple viewer initialized.")

    def is_available(self):
        return self.is_running

    def chunks_updated(self):
        return self.chunks_updated_flag

    def update_chunks(self):
        self.chunks_updated_flag = True

    def create_point_cloud_from_mesh(self, mesh):
        """Create Open3D point cloud from completed mesh"""
        try:
            print(f"Creating point cloud from completed mesh: {type(mesh)}")
            if mesh is None or not hasattr(mesh, 'vertices') or mesh.vertices is None:
                print("Error: Mesh is None or does not have vertices.")
                return None

            vertices = mesh.vertices
            if isinstance(vertices, sl.Mat):
                vertices = vertices.get_data()

            if vertices is not None and len(vertices) > 0:
                print(f"Creating point cloud from {len(vertices)} vertices.")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vertices)

                if len(vertices) > 30:
                    try:
                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                        print("Normals estimated.")
                    except Exception as e:
                        print(f"Warning: Could not estimate normals: {e}")

                if len(vertices) > 50:
                    try:
                        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                        print("Outliers removed.")
                    except Exception as e:
                        print(f"Warning: Could not remove outliers: {e}")

                self.point_cloud_o3d = pcd
                print(f"Point cloud created successfully with {len(pcd.points)} points.")
                return pcd
            else:
                print("Error: No vertices found in mesh.")
                return None
        except Exception as e:
            print(f"Error creating point cloud: {e}")
            traceback.print_exc()
            return None

    def show_open3d_visualization(self):
        """Show Open3D visualization in a separate window"""
        if self.point_cloud_o3d is not None and len(self.point_cloud_o3d.points) > 0:
            print("\nOpening Open3D visualization window...")
            print("Controls: Mouse to rotate, Scroll to zoom, Shift+Mouse to pan")
            print("Press 'Q' or 'ESC' in the 3D window to close it.")
            o3d.visualization.draw_geometries([self.point_cloud_o3d], window_name="ZED 3D Point Cloud", width=1280, height=720)
        else:
            print("\nNo point cloud available for visualization.")
            print("Please complete mapping first (Press SPACE to start/stop mapping).")

    def create_3d_visualization(self, mesh):
        """Create a 3D visualization of the mesh using matplotlib"""
        if mesh is None or not hasattr(mesh, 'vertices') or mesh.vertices is None:
            print("Mesh is None in create_3d_visualization")
            return None

        try:
            vertices = mesh.vertices
            if isinstance(vertices, sl.Mat):
                vertices = vertices.get_data()

            if vertices is not None and len(vertices) > 0:
                print(f"Creating 3D plot with {len(vertices)} vertices...")
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                sample_size = min(len(vertices), 5000)
                indices = np.random.choice(len(vertices), sample_size, replace=False)
                vertices_sample = vertices[indices]

                ax.scatter(vertices_sample[:, 0], vertices_sample[:, 1], vertices_sample[:, 2], c='blue', marker='o', s=1, alpha=0.6)
                ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
                ax.set_title(f'3D Spatial Mapping - {len(vertices)} vertices')
                
                max_range = np.array([vertices_sample[:, 0].max()-vertices_sample[:, 0].min(), 
                                      vertices_sample[:, 1].max()-vertices_sample[:, 1].min(), 
                                      vertices_sample[:, 2].max()-vertices_sample[:, 2].min()]).max() / 2.0
                mid_x = (vertices_sample[:, 0].max()+vertices_sample[:, 0].min()) * 0.5
                mid_y = (vertices_sample[:, 1].max()+vertices_sample[:, 1].min()) * 0.5
                mid_z = (vertices_sample[:, 2].max()+vertices_sample[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

                plt.savefig('temp_3d_plot.png', dpi=100)
                plt.close(fig)

                img = cv2.imread('temp_3d_plot.png')
                return cv2.resize(img, (800, 600)) if img is not None else None
            else:
                print("Vertices data is empty.")
        except Exception as e:
            print(f"Error creating 3D visualization: {e}")
            traceback.print_exc()

        img = np.zeros((480, 600, 3), dtype=np.uint8)
        cv2.putText(img, "No Vertices to Display", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img

    def update_view(self, image, pose_data, tracking_state, mapping_state, mapping_active):
        """Returns True if user wants to toggle mapping state"""
        img_display = image.get_data()
        if img_display.shape[2] == 4:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_BGRA2BGR)
        
        # Display status
        cv2.putText(img_display, f"Tracking: {tracking_state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_display, f"Mapping: {mapping_state}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if mapping_active:
            cv2.putText(img_display, "MAPPING ACTIVE - Press SPACE to stop & process", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif self.mapping_completed:
            cv2.putText(img_display, "MAPPING COMPLETED. Press SPACE to start new mapping", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img_display, "Press SPACE to start mapping", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        cv2.putText(img_display, "Press 'o' for Open3D view (after mapping)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img_display, "Press '3' for Matplotlib view (after mapping)", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(img_display, "Press 'q' to quit", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cv2.imshow("ZED Spatial Mapping", img_display)

        if self.show_3d_window:
            mesh_viz = self.create_3d_visualization(self.mesh_data) if self.mesh_data else None
            if mesh_viz is not None:
                cv2.imshow("3D Mesh Visualization (Matplotlib)", mesh_viz)
            else:
                no_mesh_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(no_mesh_img, "No mesh data available yet.", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(no_mesh_img, "Complete mapping first.", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("3D Mesh Visualization (Matplotlib)", no_mesh_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.is_running = False
        elif key == ord(' '):
            self.toggle_mapping_flag = True
        elif key == ord('3'):
            self.show_3d_window = not self.show_3d_window
            if not self.show_3d_window:
                cv2.destroyWindow("3D Mesh Visualization (Matplotlib)")
        elif key == ord('o'):
            if self.point_cloud_o3d:
                threading.Thread(target=self.show_open3d_visualization, daemon=True).start()
            else:
                print("Point cloud not ready. Complete mapping first.")

    def clear_current_mesh(self):
        print("Clearing previous mesh and point cloud data.")
        self.mesh_data = None
        self.point_cloud_o3d = None
        self.mapping_completed = False

    def close(self):
        self.is_running = False
        cv2.destroyAllWindows()


def main(opt):
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = 10.0
    parse_args(init, opt)

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        return

    # --- Positional Tracking ---
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_floor_as_origin = True
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to enable positional tracking: {returned_state}")
        zed.close()
        return

    # --- Spatial Mapping ---
    if opt.build_mesh:
        map_type = sl.SPATIAL_MAP_TYPE.MESH
        pymesh = sl.Mesh()
    else:
        map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
        pymesh = sl.FusedPointCloud()

    spatial_mapping_parameters = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.MEDIUM,
        mapping_range=sl.MAPPING_RANGE.MEDIUM,
        max_memory_usage=2048,
        save_texture=False,
        map_type=map_type
    )

    # --- Runtime variables ---
    image = sl.Mat()
    pose = sl.Pose()
    runtime_parameters = sl.RuntimeParameters(confidence_threshold=50)
    
    viewer = SimpleViewer()
    viewer.init(zed.get_camera_information().camera_configuration.calibration_parameters.left_cam, pymesh, opt.build_mesh)

    mapping_active = False
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
    last_map_request = time.time()
    
    print("\n-----------------------------------------")
    print("ZED Spatial Mapping Controls:")
    print("  [SPACE] : Start / Stop and Process Mapping")
    print("  [o]     : Show 3D Point Cloud (after mapping)")
    print("  [3]     : Show Matplotlib 3D plot (after mapping)")
    print("  [q]     : Quit")
    print("-----------------------------------------\n")


    while viewer.is_available():
        # *** CORE CHANGE: Only proceed if a new frame is successfully grabbed ***
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            
            # Retrieve data from ZED
            zed.retrieve_image(image, sl.VIEW.LEFT)
            tracking_state = zed.get_position(pose)

            if mapping_active:
                mapping_state = zed.get_spatial_mapping_state()
                # Request a map update periodically
                if time.time() - last_map_request > 0.5:
                    zed.request_spatial_map_async()
                    last_map_request = time.time()

            # Update the viewer and check for user input
            viewer.update_view(image, pose.pose_data(), tracking_state, mapping_state, mapping_active)

            # Check if the user pressed SPACE to toggle mapping
            if viewer.toggle_mapping_flag:
                viewer.toggle_mapping_flag = False  # Reset flag
                mapping_active = not mapping_active

                if mapping_active:
                    # --- START MAPPING ---
                    print("\n[INFO] Starting a new mapping session...")
                    viewer.clear_current_mesh()
                    pymesh.clear()
                    
                    # Reset tracking to define a new origin
                    zed.reset_positional_tracking(sl.Transform())
                    
                    # Enable mapping
                    err = zed.enable_spatial_mapping(spatial_mapping_parameters)
                    if err != sl.ERROR_CODE.SUCCESS:
                        print(f"Failed to enable spatial mapping: {err}")
                        mapping_active = False # Revert state
                    else:
                        last_map_request = time.time()
                        print("[INFO] Mapping started. Move the camera around.")
                else:
                    # --- STOP MAPPING and PROCESS ---
                    print("\n[INFO] Stopping mapping and processing the full map...")
                    zed.extract_whole_spatial_map(pymesh)

                    if opt.build_mesh:
                        filter_params = sl.MeshFilterParameters()
                        filter_params.set(sl.MESH_FILTER.MEDIUM)
                        pymesh.filter(filter_params, True)

                    # Check if the generated mesh/cloud is valid
                    if pymesh.vertices is not None and len(pymesh.vertices) > 0:
                        print(f"[SUCCESS] Map extracted with {len(pymesh.vertices)} vertices.")
                        viewer.mesh_data = pymesh
                        viewer.mapping_completed = True
                        
                        print("[INFO] Creating Open3D point cloud for visualization...")
                        viewer.create_point_cloud_from_mesh(pymesh)
                        
                        filepath = "zed_mesh.obj"
                        if pymesh.save(filepath):
                            print(f"[SUCCESS] Mesh saved to {filepath}")
                        else:
                            print(f"[ERROR] Failed to save the mesh.")
                    else:
                        print("[ERROR] Failed to extract a valid map. No vertices found.")
                    
                    # Disable mapping until user starts it again
                    zed.disable_spatial_mapping()
                    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    # --- Cleanup ---
    print("\n[INFO] Closing down...")
    viewer.close()
    
    # Free ZED resources
    image.free()
    pymesh.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
    # Clean up temporary files
    if os.path.exists('temp_3d_plot.png'):
        try:
            os.remove('temp_3d_plot.png')
        except OSError as e:
            print(f"Error removing temp file: {e}")

    print("[INFO] Program finished.")


def parse_args(init, opt):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print(f"[Sample] Using SVO File input: {opt.input_svo_file}")
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ':' in ip_str:
            ip, port = ip_str.split(':')
            init.set_from_stream(ip, int(port))
        else:
            init.set_from_stream(ip_str)
        print(f"[Sample] Using Stream input, IP: {ip_str}")
    
    res_dict = {
        "HD2K": sl.RESOLUTION.HD2K, "HD1200": sl.RESOLUTION.HD1200, 
        "HD1080": sl.RESOLUTION.HD1080, "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA, "VGA": sl.RESOLUTION.VGA
    }
    if opt.resolution in res_dict:
        init.camera_resolution = res_dict[opt.resolution]
        print(f"[Sample] Using Camera in resolution {opt.resolution}")
    else:
        print("[Sample] Using default resolution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it', default='')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d', default='')
    parser.add_argument('--resolution', type=str, help='Resolution, can be HD2K, HD1080, HD720, etc.', default='')
    parser.add_argument('--build_mesh', help='Generate a mesh instead of a fused point cloud', action='store_true')
    opt = parser.parse_args()
    
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only SVO file or IP address, not both. Exiting.")
        exit()
        
    main(opt)