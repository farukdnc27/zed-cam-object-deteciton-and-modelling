import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# 1. MediaPipe El Tespit Modelini Başlatma
mp_hands = mp.solutions.hands
# Modeli daha hassas ayarlayalım
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 2. RealSense Kamerasını Başlatma
pipeline = rs.pipeline()
config = rs.config()

# Daha iyi görüntü için çözünürlüğü artıralım
W, H = 1280, 720
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

pipeline.start(config)
print("RealSense kamerası başlatıldı. Çıkmak için 'q' tuşuna basın.")

# Parmak uçlarının ID'leri (MediaPipe'a göre)
tip_ids = [4, 8, 12, 16, 20] # Başparmak, İşaret, Orta, Yüzük, Serçe

try:
    while True:
        # Kameradan frame al
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Frame'i NumPy array'ine dönüştür
        image = np.asanyarray(color_frame.get_data())
        # Görüntüyü ayna gibi yansıt
        image = cv2.flip(image, 1)

        # MediaPipe için BGR'den RGB'ye dönüştür
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # El tespiti yap
        results = hands.process(image_rgb)

        # Ekranda el tespit edildiyse
        if results.multi_hand_landmarks:
            # Sadece ilk tespit edilen eli al
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # El iskeletini çiz (isteğe bağlı)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Hangi parmakların yukarıda olduğunu tutacak liste
            fingers_up = []
            
            # --- Parmak Sayma Mantığı ---
            
            # 1. Başparmak (X eksenine göre kontrol)
            # Başparmak ucu (4), bir alt ekleminden (3) daha soldaysa (sağ el için) yukarıda sayılır
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            # 2. Diğer Dört Parmak (Y eksenine göre kontrol)
            for id in range(1, 5):
                # Parmak ucu (örn: 8), iki altındaki eklemden (örn: 6) daha yukarıdaysa (Y değeri daha küçükse)
                if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            
            # Yukarıda olan toplam parmak sayısı
            total_fingers = fingers_up.count(1)
            
            # Mesajı belirle
            message = ""
            if total_fingers == 1:
                message = "Selam"
            elif total_fingers == 2:
                message = "nasilsin?"
            elif total_fingers == 3:
                message = "ben omer!"
            # Diğer durumlar için de eklemeler yapabilirsiniz
            elif total_fingers == 4:
                 message = "omer faruk dincoglu!"
            elif total_fingers == 5:
                 message = "Harika!"
                
            # Belirlenen mesajı ekrana yazdır
            # Dikdörtgen bir arkaplan ekleyerek metnin daha okunaklı olmasını sağlayalım
            cv2.rectangle(image, (30, 30), (450, 120), (0, 0, 0), cv2.FILLED) # Siyah arka plan
            cv2.putText(image, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)


        # Sonucu ekranda göster
        cv2.imshow("RealSense Jest Tanıma", image)
        
        # 'q' ile çıkış
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Kaynakları serbest bırak
    print("Kamera kapatılıyor...")
    pipeline.stop()
    cv2.destroyAllWindows()