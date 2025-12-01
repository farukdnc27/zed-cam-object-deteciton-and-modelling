import cv2
import numpy as np
import pyrealsense2 as rs

# Tıklanan noktanın bilgilerini saklamak için global değişkenler
clicked_point = None
distance_str = ""

# Fare tıklama olayını yakalayan fonksiyon
def show_distance(event, x, y, flags, param):
    global clicked_point, distance_str
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        # depth_frame'i param olarak fonksiyona iletiyoruz
        depth_frame = param
        if depth_frame:
            # Tıklanan pikselin mesafesini al (metre cinsinden)
            distance_m = depth_frame.get_distance(x, y)
            # Metre'yi santimetreye çevir ve formatla
            distance_cm = distance_m * 100
            
            if distance_cm > 0:
                distance_str = f"{distance_cm:.2f} cm"
            else:
                distance_str = "Mesafe olculemedi"
        
# --- Ana Program ---
# RealSense Pipeline ve Config
pipeline = rs.pipeline()
config = rs.config()

# Renk ve Derinlik akışlarını etkinleştir
W, H = 1280, 720
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# Pipeline'ı başlat
profile = pipeline.start(config)

# Derinlik sensörünün ölçeğini al (mesafe hesaplaması için)
# depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# Derinlik ve Renk karelerini hizalamak için bir align objesi oluştur
align_to = rs.stream.color
align = rs.align(align_to)

print("Kamera başlatıldı. Uzaklığını ölçmek istediğiniz yere tıklayın.")
print("Çıkmak için 'q' tuşuna basın.")

# Pencereyi oluştur ve fare olayını bağla
cv2.namedWindow('RealSense Mesafe Olcer')

try:
    while True:
        # Frame'leri bekle
        frames = pipeline.wait_for_frames()
        
        # Frame'leri hizala
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue
        
        # OpenCV'nin kullanabilmesi için görüntüleri NumPy array'ine dönüştür
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Fare olayını yakalamak için derinlik çerçevesini parametre olarak ata
        cv2.setMouseCallback('RealSense Mesafe Olcer', show_distance, aligned_depth_frame)
        
        # Eğer bir noktaya tıklandıysa, o noktayı ve mesafeyi göster
        if clicked_point:
            # Tıklanan yere bir daire çiz
            cv2.circle(color_image, clicked_point, 5, (0, 0, 255), -1)
            # Mesafeyi yazdır
            cv2.putText(color_image, distance_str, (clicked_point[0] + 10, clicked_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        # Görüntüyü göster
        cv2.imshow('RealSense Mesafe Olcer', color_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()