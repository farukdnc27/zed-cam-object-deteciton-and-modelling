import pyrealsense2 as rs
import numpy as np
import cv2

# --- Global Değişkenler ---
points_to_measure = []
measuring = False
# YENİ: Ölçümün tamamlanıp tamamlanmadığını ve sonucunu saklamak için değişkenler
measurement_complete = False
last_distance = 0.0

# --- Fare Tıklamalarını Yakalamak İçin Fonksiyon ---
def mouse_callback(event, x, y, flags, param):
    """
    Bu fonksiyon, fare tıklamalarını yakalar ve ölçüm için noktaları belirler.
    Eğer bir ölçüm tamamlanmışsa yeni tıklamaları engeller.
    """
    global points_to_measure, measuring, measurement_complete
    
    # Eğer ekranda zaten bir ölçüm sonucu varsa, yeni nokta seçilmesini engelle
    if measurement_complete:
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        if not measuring:
            points_to_measure = [(x, y)]
            measuring = True
            print(f"Başlangıç noktası seçildi: ({x}, {y})")
        else:
            points_to_measure.append((x, y))
            measuring = False
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
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)

# --- Ana Döngü ---
cv2.namedWindow('RealSense Measurement', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('RealSense Measurement', mouse_callback)
print("\nÖlçüm için iki nokta seçin.")
print("Yeni bir ölçüm yapmak için SPACE tuşuna basın.")
print("Çıkmak için 'q' veya 'ESC' tuşuna basın.\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
            
        filtered_depth_frame = spatial.process(depth_frame)
        filtered_depth_frame = temporal.process(filtered_depth_frame)
        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Ölçüm yapılıyorsa (ilk tık yapılmışsa), imleci göster
        if measuring and points_to_measure:
            cv2.circle(color_image_bgr, points_to_measure[0], 4, (0, 0, 255), -1)

        # --- YENİ MANTIK ---
        # 1. Adım: Ölçümün yeni tamamlanıp tamamlanmadığını kontrol et
        if len(points_to_measure) == 2 and not measurement_complete:
            p1 = points_to_measure[0]
            p2 = points_to_measure[1]
            
            depth_raw_p1 = depth_image[p1[1], p1[0]]
            depth_raw_p2 = depth_image[p2[1], p2[0]]
            depth_m_p1 = depth_raw_p1 * depth_scale
            depth_m_p2 = depth_raw_p2 * depth_scale
            
            # Eğer noktalardan biri için derinlik okunamadıysa (değer 0 ise) ölçümü iptal et
            if depth_m_p1 == 0 or depth_m_p2 == 0:
                print("Hata: Seçilen noktalardan biri için derinlik okunamadı. Tekrar deneyin.")
                points_to_measure = []
                measuring = False
            else:
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                point1_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [p1[0], p1[1]], depth_m_p1)
                point2_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [p2[0], p2[1]], depth_m_p2)
                
                # Mesafeyi hesapla ve global değişkene kaydet
                last_distance = np.sqrt(
                    (point1_3d[0] - point2_3d[0])**2 +
                    (point1_3d[1] - point2_3d[1])**2 +
                    (point1_3d[2] - point2_3d[2])**2
                )
                
                # Ölçümün tamamlandığını belirten bayrağı ayarla
                measurement_complete = True
        
        # 2. Adım: Eğer bir ölçüm tamamlandıysa, sonucu her karede ekrana çiz
        if measurement_complete:
            p1 = points_to_measure[0]
            p2 = points_to_measure[1]
            cv2.line(color_image_bgr, p1, p2, (0, 255, 0), 2)
            cv2.circle(color_image_bgr, p1, 4, (0, 0, 255), -1)
            cv2.circle(color_image_bgr, p2, 4, (0, 0, 255), -1)
            cv2.putText(color_image_bgr, f"{last_distance * 1000:.2f} mm", 
                        (p1[0] + 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Görüntüyü göster
        cv2.imshow('RealSense Measurement', color_image_bgr)
        key = cv2.waitKey(1)
        
        # 'q' veya 'ESC' tuşuna basıldığında çık
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        # YENİ: SPACE tuşuna basıldığında ölçümü sıfırla
        elif key == ord(' '): # ord(' ') boşluk tuşunun kodunu verir
            print("Ölçüm sıfırlandı. Yeni noktalar seçebilirsiniz.")
            points_to_measure = []
            measuring = False
            measurement_complete = False

finally:
    pipeline.stop()