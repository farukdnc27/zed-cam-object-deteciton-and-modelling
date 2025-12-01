import cv2
import numpy as np
import mediapipe as mp

# --- YENİ: KAMERA SEÇİMİ ---
# Kullanmak istediğiniz kamerayı buradan seçin:
# 'realsense' -> Intel RealSense Kamera
# 'webcam'    -> Bilgisayarın varsayılan kamerası (laptop kamerası, USB webcam vb.)
CAMERA_CHOICE = 'webcam'  # <-- BU SATIRI DEĞİŞTİREREK SEÇİM YAPIN

# --- YENİ: Koşullu Kütüphane Yükleme ---
# Sadece gerekliyse pyrealsense2'yi yüklemeye çalışalım.
if CAMERA_CHOICE == 'realsense':
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("Hata: pyrealsense2 kütüphanesi bulunamadı.")
        print("Lütfen 'pip install pyrealsense2' komutuyla yükleyin veya CAMERA_CHOICE'u 'webcam' yapın.")
        exit()

# MediaPipe el takip modelini başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# --- DEĞİŞTİ: Kamera Başlatma Bloğu ---
if CAMERA_CHOICE == 'realsense':
    print("Intel RealSense kamera başlatılıyor...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("RealSense kamera başlatıldı.")
elif CAMERA_CHOICE == 'webcam':
    print("Varsayılan webcam başlatılıyor...")
    # 0, genellikle varsayılan webcam'dir. Eğer harici bir USB kameranız varsa 1, 2 gibi değerleri deneyebilirsiniz.
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Hata: Webcam açılamadı. Başka bir program tarafından kullanılıyor olabilir mi?")
        exit()
    print("Webcam başlatıldı.")
else:
    print(f"Hata: Geçersiz kamera seçimi: '{CAMERA_CHOICE}'. Lütfen 'realsense' veya 'webcam' seçin.")
    exit()

print("Çıkmak için 'q' tuşuna basın.")

try:
    while True:
        # --- DEĞİŞTİ: Görüntü Alma Bloğu ---
        # Seçime göre ilgili kameradan görüntü al
        if CAMERA_CHOICE == 'realsense':
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
        else: # webcam
            success, color_image = cap.read()
            if not success:
                print("Webcam'den görüntü alınamadı. Döngü sonlandırılıyor.")
                break
            # Webcam'den gelen görüntü bazen ters olabilir, isterseniz düzeltebilirsiniz:
            # color_image = cv2.flip(color_image, 1) 

        # --- Bundan sonraki kod her iki kamera için de ortaktır ---

        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        gesture_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                fingers_up = []

                # Parmakların açık/kapalı durumunu kontrol et
                # İşaret parmağı
                if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                    fingers_up.append(True)
                else:
                    fingers_up.append(False)
                # Orta parmak
                if landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                    fingers_up.append(True)
                else:
                    fingers_up.append(False)
                # Yüzük parmağı
                if landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y:
                    fingers_up.append(True)
                else:
                    fingers_up.append(False)
                # Serçe parmak
                if landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y:
                    fingers_up.append(True)
                else:
                    fingers_up.append(False)

                # Hareketleri tanımla
                if fingers_up == [True, False, False, False]:
                    gesture_text = "Merhaba"
                elif fingers_up == [True, True, False, False]:
                    gesture_text = "Nasılsın"

        # Metni ekrana yazdır
        cv2.putText(image_bgr, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('El Hareketi Tanıma', image_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    # --- DEĞİŞTİ: Temizleme Bloğu ---
    print("Program kapatılıyor.")
    if CAMERA_CHOICE == 'realsense':
        pipeline.stop()
    else: # webcam
        cap.release()
    
    hands.close()
    cv2.destroyAllWindows()