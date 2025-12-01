import open3d as o3d
import pyrealsense2 as rs
import numpy as np
import cv2  # OpenCV kütüphanesi burada import ediliyor.

# --- Konfigürasyon ---
pipeline = rs.pipeline()
config = rs.config()
W, H = 640, 480
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# --- Open3D Ayarları ---
vis = o3d.visualization.Visualizer()
vis.create_window("Oda Haritası", width=W, height=H)

voxel_length = 0.02 # Daha hızlı performans için voksel boyutunu biraz artırdım.
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_length,
    sdf_trunc=voxel_length * 3,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
)

camera_pose = np.identity(4)
is_first_frame = True

print("Kamera başlatıldı. Odanın haritasını çıkarmak için kamerayı YAVAŞÇA hareket ettirin.")
print("3D GÖRSELLEŞTİRME PENCERESİNDE harita oluşmasını izleyin.")
print("İşiniz bitince terminale dönüp CTRL+C'ye basın.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
            
        depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        color_image = o3d.geometry.Image(np.asanyarray(cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, 
            depth_image, 
            depth_scale=profile.get_device().first_depth_sensor().get_depth_scale(),
            depth_trunc=4.0, 
            convert_rgb_to_intensity=False
        )

        volume.integrate(rgbd_image, o3d_intrinsics, np.linalg.inv(camera_pose))
        pcd = volume.extract_point_cloud()
        
        if is_first_frame:
            vis.add_geometry(pcd)
            is_first_frame = False
        else:
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        
except KeyboardInterrupt:
    print("Kullanıcı tarafından durduruldu.")
except Exception as e:
    print("Bir hata oluştu:", e)

finally:
    print("Haritalama tamamlandı. Sonuç kaydediliyor...")
    
    pipeline.stop()
    vis.destroy_window()
    
    final_pcd = volume.extract_point_cloud()
    
    # Eğer nokta bulutu boş değilse kaydet ve temizle
    if not final_pcd.has_points():
        print("Hiç nokta bulutu verisi toplanmadı. Dosya oluşturulmayacak.")
    else:
        final_pcd_downsampled = final_pcd.voxel_down_sample(voxel_size=voxel_length)
        
        # DÜZELTİLMİŞ FONKSİYON ADI: '_rejection' kısmı kaldırıldı.
        final_pcd_clean, _ = final_pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        o3d.io.write_point_cloud("oda_haritasi.ply", final_pcd_clean)
        print("Nokta bulutu 'oda_haritasi.ply' olarak kaydedildi.")