import pyrealsense2 as rs
import numpy as np
import cv2

# Pipeline'ı yapılandır
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Kamera başlatılıyor...")
pipeline.start(config)

# Derinlik haritasını renklendirmek için bir obje
colorizer = rs.colorizer()

try:
    while True:
        # Frame'leri bekle
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Derinlik verisini insan gözünün görebileceği bir renk haritasına dönüştür
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        
        # Normal renkli görüntüyü al
        color_image = np.asanyarray(color_frame.get_data())

        # İki görüntüyü yan yana göster
        images = np.hstack((color_image, depth_color_image))

        cv2.imshow('RealSense - Renkli ve Derinlik', images)
        
        # 'q' ile çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()