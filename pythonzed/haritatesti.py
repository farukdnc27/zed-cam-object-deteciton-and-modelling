import open3d as o3d
import pyrealsense2 as rs
import numpy as np

print("Son test başlatılıyor. Daha temel bir fonksiyon denenecek...")

pipeline = rs.pipeline()
config = rs.config()
W, H = 640, 480
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
    
    # Kameranın ısınması için bekle
    for i in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        raise RuntimeError("Gerekli kareler alınamadı.")

    # --- YENİ VE GÜVENİLİR YÖNTEM ---
    # Sadece derinlik verisini Open3D formatına çevir
    depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
    
    # Kamera parametrelerini al
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    )

    # Sadece derinlikten nokta bulutu oluştur
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        o3d_intrinsics,
        depth_scale=profile.get_device().first_depth_sensor().get_depth_scale(),
        depth_trunc=4.0 # 4 metreden uzak noktaları yoksay
    )

    print(f"\n--- SONUÇ ---")
    print(f"Oluşturulan nokta sayısı: {len(pcd.points)}")

    if not pcd.has_points():
        print("HATA: Bu temel yöntem bile başarısız oldu. Sorun çok daha derinde olabilir.")
    else:
        print("!!! BAŞARILI !!! Nokta bulutu oluşturuldu!")
        print("Görselleştirme penceresi açılıyor...")
        
        # Nokta bulutunu görselleştir. Bu sefer renkli olmasa da 3D bir şekil görmeliyiz.
        o3d.visualization.draw_geometries([pcd], window_name="BAŞARILI TEST")

except Exception as e:
    print(f"Bir hata oluştu: {e}")

finally:
    pipeline.stop()
    print("Test tamamlandı.")