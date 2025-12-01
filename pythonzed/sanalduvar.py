# Gerekli kütüphaneleri içeri aktar
import cv2
import pyrealsense2 as rs
import numpy as np

# --- 1. AYARLAR VE GLOBAL DEĞİŞKENLER ---

# Takip edilecek nesnenin kutusunu (Region of Interest - ROI) saklamak için
# Kullanıcı fare ile bir alan seçtiğinde bu değişken doldurulacak.
roi_box = None

# OpenCV'nin nesne takipçisini (tracker) saklamak için
# Bir nesne seçildiğinde bu değişkene bir takip algoritması atanacak.
tracker = None

# Fare ile çizim yapılıp yapılmadığını ve başlangıç koordinatlarını kontrol eden değişkenler
drawing = False
ix, iy = -1, -1

# Fare olaylarını (tıklama, sürükleme, bırakma) yöneten fonksiyon
def select_roi(event, x, y, flags, param):
    """
    Bu fonksiyon, OpenCV penceresindeki fare hareketlerini dinler.
    Kullanıcının bir dikdörtgen çizerek nesne seçmesini sağlar.
    """
    # Global değişkenlere erişeceğimizi belirtiyoruz
    global ix, iy, drawing, roi_box, frame

    # Farenin sol tuşuna basıldığında
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True  # Çizim modunu aktif et
        ix, iy = x, y   # Başlangıç koordinatlarını kaydet
        roi_box = None  # Yeni bir seçim yapılacağı için eski seçimi temizle

    # Fare basılıyken hareket ettiğinde
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # Anlık olarak bir dikdörtgen çizerek kullanıcıya geri bildirim ver
            # Orijinal frame'in kopyası üzerinde çizim yapıyoruz ki ana görüntü bozulmasın
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            # Geçici çizimi göstermek için imshow'u burada çağırıyoruz
            cv2.imshow("Nesne Takibi", frame_copy)

    # Farenin sol tuşu bırakıldığında
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False  # Çizim modunu kapat
        
        # Dikdörtgenin genişlik ve yüksekliğini hesapla
        w = abs(x - ix)
        h = abs(y - iy)
        
        # Çok küçük bir alan seçildiyse bunu yok say
        if w > 10 and h > 10:
            # Geçerli bir seçim yapıldı, bu kutuyu roi_box olarak kaydet
            roi_box = (min(ix, x), min(iy, y), w, h)


# --- 2. KAMERA BAŞLATMA VE HİZALAMA ---

print("Kamera başlatılıyor...")
# RealSense pipeline'ı, kamera ile olan ana iletişim kanalımızdır
pipeline = rs.pipeline()
config = rs.config()

# Hem renk hem de derinlik akışını etkinleştiriyoruz. Bu proje için ikisi de gerekli.
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Yapılandırmayı başlat
pipeline.start(config)

# KRİTİK ADIM: HİZALAMA (ALIGNMENT)
# Renk ve derinlik sensörleri fiziksel olarak farklı yerlerde olduğu için,
# aynı (x, y) pikseli farklı noktalara denk gelir.
# 'align' objesi, derinlik haritasını renkli görüntüyle eşleştirir.
align_to = rs.stream.color
align = rs.align(align_to)

# OpenCV penceresi oluştur ve fare olaylarını yönetecek fonksiyonumuzu bu pencereye bağla
cv2.namedWindow("Nesne Takibi")
cv2.setMouseCallback("Nesne Takibi", select_roi)

print("Kamera başlatıldı. Takip edilecek nesneyi fare ile seçin.")
print("Yeni bir nesne seçmek için tekrar çizin. Çıkmak için 'q' basın.")


# --- 3. ANA DÖNGÜ ---
try:
    while True:
        # Kameradan yeni frame'lerin gelmesini bekle
        frames = pipeline.wait_for_frames()
        # Gelen frame'leri renk kamerasıyla hizala
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Frame'lerin başarıyla alınıp alınmadığını kontrol et
        if not depth_frame or not color_frame:
            continue

        # Frame verilerini OpenCV'nin kullanabileceği NumPy dizilerine dönüştür
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())

        # Eğer bir takipçi (tracker) aktif ise
        if tracker is not None:
            # Takipçiyi yeni frame ile güncelle
            success, box = tracker.update(frame)

            if success:
                # Takip başarılıysa, nesnenin yeni konumunu al
                (x, y, w, h) = [int(v) for v in box]
                # Ekrana nesnenin etrafına bir kutu çiz
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # --- UZAKLIK ÖLÇÜMÜ ---
                # Takip edilen kutunun içindeki derinlik bölgesini (ROI) al
                depth_roi = depth_image[y:y+h, x:x+w]
                # Geçersiz derinlik değerlerini (0 olanları) filtrele
                non_zero_depths = depth_roi[depth_roi > 0]
                
                if non_zero_depths.size > 0:
                    # Ortalama yerine MEDYAN kullanmak, aykırı değerlere karşı daha sağlamdır
                    distance_mm = np.median(non_zero_depths)
                    distance_m = distance_mm / 1000.0
                    
                    # Hesaplanan uzaklığı ekrana yazdır
                    distance_text = f"Uzaklik: {distance_m:.2f} m"
                    cv2.putText(frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Takip başarısız olduysa (nesne kaybolduysa)
                cv2.putText(frame, "Takip Basarisiz!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                tracker = None # Takipçiyi sıfırla ki yeni seçim yapılabilsin

        # Eğer kullanıcı fare ile yeni bir nesne seçtiyse
        if roi_box is not None:
            # --- DAHA SAĞLAM TAKİPÇİ OLUŞTURMA ---
            try:
                # Standart yöntem (opencv-contrib-python gerektirir)
                tracker = cv2.TrackerCSRT_create()
            except AttributeError:
                # Bazı eski OpenCV 4.x versiyonları için yedek yöntem (legacy modülü)
                try:
                    tracker = cv2.legacy.TrackerCSRT_create()
                except AttributeError:
                    print("\nHATA: OpenCV takip modülleri bulunamadı!")
                    print("Lütfen 'pip install opencv-contrib-python' komutunu çalıştırdığınızdan emin olun.")
                    print("Program sonlandırılıyor.")
                    break # Hata durumunda döngüden çık
            # --- ---

            # Takipçi başarıyla oluşturulduysa başlat
            tracker.init(frame, roi_box)
            roi_box = None # Seçim kutusunu sıfırla ki bu blok tekrar tekrar çalışmasın
            print("Yeni nesne seçildi ve takip başlatıldı.")
        
        # Eğer aktif bir takipçi yoksa (başlangıçta veya takip kaybedildiğinde)
        elif tracker is None:
            # Kullanıcıya ne yapması gerektiğini söyleyen bir mesaj göster
            cv2.putText(frame, "Takip edilecek nesneyi secin", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 150, 255), 2)

        # Sonuç görüntüsünü ekranda göster
        # Not: Fare ile çizim yapılırken bu satır çalışmaz, çünkü imshow
        #      select_roi fonksiyonunun içinde çağrılır.
        if not drawing:
            cv2.imshow("Nesne Takibi", frame)

        # 'q' tuşuna basılırsa döngüden çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
finally:
    # --- 4. TEMİZLİK ---
    # Program kapatılırken tüm kaynakları serbest bırak
    print("Kapatılıyor.")
    pipeline.stop()
    cv2.destroyAllWindows()