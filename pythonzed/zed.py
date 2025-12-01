import pyrealsense2 as rs
import numpy as np
import cv2

# Pipeline ve config nesnelerini oluştur
pipeline = rs.pipeline()
config = rs.config()

# Akışları yapılandır
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.accel)  # İvmeölçer
config.enable_stream(rs.stream.gyro)   # Jiroskop

# Akışı başlat
pipeline.start(config)

try:
    while True:
        # Çerçeveleri bekle
        frames = pipeline.wait_for_frames()
        
        # Renk ve derinlik çerçevelerini al
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        # IMU verilerini kontrol et
        if frames.size() >= 3:
            for f in frames:
                if f.is_motion_frame():
                    motion_data = f.as_motion_frame().get_motion_data()
                    if f.profile.stream_type() == rs.stream.accel:
                        print(f"Accel: X={motion_data.x:.3f}, Y={motion_data.y:.3f}, Z={motion_data.z:.3f}")
                    elif f.profile.stream_type() == rs.stream.gyro:
                        print(f"Gyro: X={motion_data.x:.3f}, Y={motion_data.y:.3f}, Z={motion_data.z:.3f}")
        
        # Renk ve derinlik görüntülerini numpy dizisine dönüştür
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Derinlik görüntüsünü renkli hale getir (görselleştirme için)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Görüntüleri göster
        cv2.imshow('Renk Goruntusu', color_image)
        cv2.imshow('Derinlik Goruntusu', depth_colormap)
        
        # Çıkış için 'q' tuşuna bas
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Akışı durdur
    pipeline.stop()
    cv2.destroyAllWindows()