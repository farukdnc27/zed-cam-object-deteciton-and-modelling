import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Önceki kodda kullandığımız DepthCamera sınıfını buraya kopyalayabilirsiniz.
# Veya daha modüler olması için ayrı bir dosyada tutup import edebilirsiniz.
class DepthCamera:
    def __init__(self, resolution_width, resolution_height):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_raw_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        return aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()

    def get_intrinsics(self):
        return self.profile.get_stream(rs.stream.video).as_video_stream_profile().get_intrinsics()

    def release(self):
        self.pipeline.stop()

def main():
    # Kamera yapılandırması
    resolution_width, resolution_height = (640, 480)
    realsense_cam = DepthCamera(resolution_width, resolution_height)
    intrinsics = realsense_cam.get_intrinsics()
    
    # Open3D görselleştiriciyi ayarla
    vis = o3d.visualization.Visualizer()
    vis.create_window("Open3D Gerçek Zamanlı Nokta Bulutu", width=1280, height=720)
    
    pcd = o3d.geometry.PointCloud()
    is_first_frame = True

    try:
        while True:
            # Kameradan kareleri al
            depth_frame, color_frame = realsense_cam.get_raw_frame()
            if not depth_frame or not color_frame:
                continue

            # Open3D için görüntüleri oluştur
            # Renkli görüntü BGR'den RGB'ye dönüştürülmeli
            color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
            depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
            
            # RGBD görüntüsü oluştur
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, 
                depth_image, 
                depth_scale=realsense_cam.depth_scale,
                depth_trunc=3.0, # 3 metreden uzak noktaları göz ardı et
                convert_rgb_to_intensity=False
            )
            
            # RGBD görüntüsünden nokta bulutu oluştur
            temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    width=intrinsics.width,
                    height=intrinsics.height,
                    fx=intrinsics.fx,
                    fy=intrinsics.fy,
                    ppx=intrinsics.ppx,
                    ppy=intrinsics.ppy
                )
            )
            
            # Nokta bulutunu güncelle
            pcd.points = temp_pcd.points
            pcd.colors = temp_pcd.colors
            
            if is_first_frame:
                vis.add_geometry(pcd)
                is_first_frame = False
            else:
                vis.update_geometry(pcd)
            
            vis.poll_events()
            vis.update_renderer()

    finally:
        realsense_cam.release()
        vis.destroy_window()
        print("Kaynaklar serbest bırakıldı.")

if __name__ == '__main__':
    main()