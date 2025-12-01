import pyrealsense2 as rs
import numpy as np
import cv2

# --- Global Değişkenler ---
# Mevcut ölçüm için geçici olarak noktaları tutan liste
current_points = []
# Tamamlanmış tüm ölçümleri saklayan ana liste
# Her eleman [nokta1, nokta2, mesafe] formatında olacak
completed_measurements = []
# Ölçümün aktif olup olmadığını belirten bayrak (1. tıklama sonrası True olur)
is_measuring = False

# --- Fare Tıklamalarını Yakalamak İçin Fonksiyon ---
def mouse_callback(event, x, y, flags, param):
    global current_points, is_measuring

    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_measuring:
            # İlk tıklama: Yeni ölçümü başlat
            current_points = [(x, y)]
            is_measuring = True
            print(f"Başlangıç noktası seçildi: ({x}, {y})")
        else:
            # İkinci tıklama: Mevcut ölçümü bitir
            current_points.append((x, y))
            is_measuring = False
            print(f"Bitiş noktası seçildi: ({x}, {y})")

# --- RealSense Yapılandırması (Değişiklik yok) ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Derinlik Ölçeği (Depth Scale): {depth_scale}")
align_to = rs.stream.color
align = rs.align(align_to)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

# --- Ana Döngü ---
cv2.namedWindow('RealSense - Çoklu Ölçüm Aracı', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('RealSense - Çoklu Ölçüm Aracı', mouse_callback)
print("\nÖlçüm için iki nokta seçin. İstediğiniz kadar ölçüm yapabilirsiniz.")
print("Tüm ölçümleri temizlemek için SPACE tuşuna basın.")
print("Çıkmak için 'q' veya 'ESC' tuşuna basın.\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
            
        filtered_depth_frame = temporal.process(spatial.process(depth_frame))
        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # --- YENİ MANTIK: Ölçüm Hesaplama ve Saklama ---
        # Eğer yeni bir ölçüm tamamlandıysa (current_points listesinde 2 nokta varsa)
        if len(current_points) == 2:
            p1 = current_points[0]
            p2 = current_points[1]
            
            depth_m_p1 = depth_image[p1[1], p1[0]] * depth_scale
            depth_m_p2 = depth_image[p2[1], p2[0]] * depth_scale
            
            if depth_m_p1 > 0 and depth_m_p2 > 0:
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                point1_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [p1[0], p1[1]], depth_m_p1)
                point2_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [p2[0], p2[1]], depth_m_p2)
                
                distance = np.sqrt(sum([(a - b) ** 2 for a, b in zip(point1_3d, point2_3d)]))
                
                # Hesaplanan ölçümü tamamlanmışlar listesine ekle
                completed_measurements.append([p1, p2, distance])
            else:
                print("Hata: Derinlik okunamadı. Ölçüm iptal edildi.")

            # Geçici listeyi temizleyerek yeni bir ölçüme hazır hale gel
            current_points = []

        # --- YENİ MANTIK: Görselleştirme ---
        
        # 1. Tamamlanmış tüm ölçümleri ekrana çiz
        if completed_measurements:
            for p1, p2, dist in completed_measurements:
                cv2.line(color_image_bgr, p1, p2, (0, 255, 0), 2)
                cv2.circle(color_image_bgr, p1, 4, (0, 0, 255), -1)
                cv2.circle(color_image_bgr, p2, 4, (0, 0, 255), -1)
                cv2.putText(color_image_bgr, f"{dist * 1000:.2f} mm", 
                            (p1[0] + 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 2. Eğer yeni bir ölçüm yapılıyorsa (ilk nokta seçilmişse), o noktayı göster
        if is_measuring and current_points:
            cv2.circle(color_image_bgr, current_points[0], 4, (0, 255, 255), -1) # Farklı renk (sarı)

        # Görüntüyü göster
        cv2.imshow('RealSense - Çoklu Ölçüm Aracı', color_image_bgr)
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        # SPACE tuşu tüm ölçümleri temizler
        elif key == ord(' '):
            print("Tüm ölçümler temizlendi.")
            completed_measurements = []
            current_points = []
            is_measuring = False

finally:
    pipeline.stop()