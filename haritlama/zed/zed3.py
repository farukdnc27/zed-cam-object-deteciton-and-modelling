import pyzed.sl as sl
import cv2
import numpy as np
import time
from datetime import datetime
import os

class ZEDCamera:
    """
    ZED Kamera için temel işlemler sınıfı
    """
    
    def __init__(self, resolution=sl.RESOLUTION.HD720, fps=30):
        """
        ZED kamerayı başlat
        
        Args:
            resolution: Çözünürlük (HD2K, HD1080, HD720, VGA)
            fps: Saniyedeki kare sayısı
        """
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.METER
        
        # Görüntü matrisleri
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.depth_map = sl.Mat()
        self.point_cloud = sl.Mat()
        
        self.is_opened = False
    
    def open_camera(self):
        """Kamerayı aç"""
        print("ZED kamera açılıyor...")
        
        # Önce kamera kontrolü yap
        if not self.check_camera_connection():
            return False
        
        err = self.zed.open(self.init_params)
        
        if err != sl.ERROR_CODE.SUCCESS:
            self.handle_camera_error(err)
            return False
        
        self.is_opened = True
        print("ZED kamera başarıyla açıldı")
        
        # Kamera bilgilerini güvenli şekilde göster
        self.show_camera_info()
        return True
    
    def check_camera_connection(self):
        """Kamera bağlantısını kontrol et"""
        cameras = sl.Camera.get_device_list()
        
        if len(cameras) == 0:
            print("HATA: ZED kamera bulunamadı!")
            print("Kontrol edilecekler:")
            print("- Kamera USB portuna bağlı mı?")
            print("- USB 3.0 port kullanılıyor mu?")
            print("- Kamera güç kaynağı yeterli mi?")
            print("- ZED SDK doğru kuruldu mu?")
            return False
        
        print(f"Bulunan kameralar: {len(cameras)}")
        for i, cam in enumerate(cameras):
            print(f"  Kamera {i}: {cam.serial_number}")
        
        return True
    
    def handle_camera_error(self, error_code):
        """Kamera hatalarını işle"""
        error_messages = {
            sl.ERROR_CODE.CAMERA_NOT_DETECTED: "Kamera algılanamadı - USB bağlantısını kontrol edin",
            sl.ERROR_CODE.SENSORS_NOT_INITIALIZED: "Sensörler başlatılamadı",
            sl.ERROR_CODE.INVALID_RESOLUTION: "Geçersiz çözünürlük",
            sl.ERROR_CODE.LOW_USB_BANDWIDTH: "Düşük USB bant genişliği - USB 3.0 kullanın",
            sl.ERROR_CODE.CAMERA_NOT_AVAILABLE: "Kamera kullanılamıyor - başka uygulama kullanıyor olabilir",
            sl.ERROR_CODE.INVALID_SVO_FILE: "Geçersiz SVO dosyası",
            sl.ERROR_CODE.SVO_RECORDING_ERROR: "SVO kayıt hatası",
            sl.ERROR_CODE.INVALID_COORDINATE_SYSTEM: "Geçersiz koordinat sistemi",
            sl.ERROR_CODE.INVALID_FIRMWARE: "Geçersiz firmware - güncelleme gerekli",
            sl.ERROR_CODE.NOT_A_NEW_FRAME: "Yeni frame yok",
            sl.ERROR_CODE.CUDA_ERROR: "CUDA hatası",
            sl.ERROR_CODE.CAMERA_REBOOTING: "Kamera yeniden başlatılıyor"
        }
        
        message = error_messages.get(error_code, f"Bilinmeyen hata: {error_code}")
        print(f"Kamera Hatası: {message}")
        
        # Öneriler
        print("\nÇözüm önerileri:")
        print("1. Kamerayı çıkarıp tekrar takın")
        print("2. USB 3.0 port kullandığınızdan emin olun")  
        print("3. Başka uygulamaları kapatın")
        print("4. ZED SDK'yı güncelleyin")
        print("5. Bilgisayarı yeniden başlatın")
    
    def show_camera_info(self):
        """Kamera bilgilerini göster"""
        if not self.is_opened:
            return
        
        try:
            cam_info = self.zed.get_camera_information()
            print(f"\nKamera Bilgileri:")
            print(f"Seri No: {cam_info.serial_number}")
            
            # Firmware bilgisi için farklı versiyonları dene
            try:
                if hasattr(cam_info, 'camera_firmware_version'):
                    print(f"Firmware: {cam_info.camera_firmware_version}")
                elif hasattr(cam_info, 'firmware_version'):
                    print(f"Firmware: {cam_info.firmware_version}")
                else:
                    print("Firmware: Bilgi alınamadı")
            except:
                print("Firmware: Bilgi alınamadı")
            
            print(f"Çözünürlük: {cam_info.camera_resolution.width}x{cam_info.camera_resolution.height}")
            print(f"FPS: {self.zed.get_init_parameters().camera_fps}")
            
            # Model bilgisi varsa göster
            if hasattr(cam_info, 'camera_model'):
                print(f"Model: {cam_info.camera_model}")
                
        except Exception as e:
            print(f"Kamera bilgileri alınırken hata: {e}")
    
    def capture_frame(self):
        """Tek frame yakala"""
        if not self.is_opened:
            print("Kamera açık değil!")
            return None, None, None
        
        runtime_params = sl.RuntimeParameters()
        
        if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Sol ve sağ görüntüleri al
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            
            # Derinlik haritasını al
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            
            # NumPy array'e çevir
            left_img = self.image_left.get_data()
            right_img = self.image_right.get_data()
            depth_img = self.depth_map.get_data()
            
            return left_img, right_img, depth_img
        
        return None, None, None
    
    def live_stream(self, show_depth=True, save_images=False):
        """Canlı görüntü akışı"""
        if not self.is_opened:
            print("Önce kamerayı açın!")
            return
        
        print("Canlı akış başladı. 'q' tuşuna basarak çıkın.")
        
        if save_images:
            save_dir = f"zed_captures_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(save_dir, exist_ok=True)
            frame_count = 0
        
        try:
            while True:
                left_img, right_img, depth_img = self.capture_frame()
                
                if left_img is not None:
                    # BGR'den RGB'ye çevir (OpenCV için)
                    left_display = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                    
                    if show_depth and depth_img is not None:
                        # Derinlik görüntüsünü normalize et
                        depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        
                        # Yan yana göster
                        combined = np.hstack((left_display, depth_colored))
                        cv2.imshow('ZED - Sol Görüntü | Derinlik', combined)
                    else:
                        cv2.imshow('ZED - Sol Görüntü', left_display)
                    
                    # Görüntü kaydetme
                    if save_images and frame_count % 30 == 0:  # Her 30 frame'de bir kaydet
                        cv2.imwrite(f"{save_dir}/left_{frame_count:06d}.jpg", left_display)
                        if depth_img is not None:
                            cv2.imwrite(f"{save_dir}/depth_{frame_count:06d}.jpg", depth_colored)
                    
                    frame_count += 1
                
                # Çıkış kontrolü
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nAkış durduruldu.")
        
    def volume_measurement(self):
        """3D hacim ölçümü"""
        print("=== HACİM ÖLÇÜMÜ MODU ===")
        print("Talimatlar:")
        print("- Sol tık: Hacim sınırları çizin")
        print("- Sağ tık: Noktaları temizle")
        print("- 'c': Hacmi hesapla")
        print("- 'r': Sıfırla")
        print("- 'q': Çıkış")
        
        points_2d = []
        volume_result = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points_2d
            
            if event == cv2.EVENT_LBUTTONDOWN:
                points_2d.append((x, y))
                print(f"Nokta eklendi: ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN:
                points_2d.clear()
                print("Noktalar temizlendi")
        
        cv2.namedWindow('Hacim Ölçümü')
        cv2.setMouseCallback('Hacim Ölçümü', mouse_callback)
        
        while True:
            left_img, _, depth_img = self.capture_frame()
            
            if left_img is not None:
                display_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                
                # Seçilen noktaları çiz
                for i, point in enumerate(points_2d):
                    cv2.circle(display_img, point, 5, (0, 255, 0), -1)
                    cv2.putText(display_img, str(i+1), 
                              (point[0]+10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Çokgen çiz
                if len(points_2d) > 2:
                    pts = np.array(points_2d, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(display_img, [pts], True, (255, 0, 0), 2)
                
                # Hacim sonucunu göster
                if volume_result is not None:
                    cv2.putText(display_img, f"Hacim: {volume_result:.4f} m³", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Durum bilgisi
                status = f"Nokta sayısı: {len(points_2d)}"
                cv2.putText(display_img, status, (10, display_img.shape[0]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Hacim Ölçümü', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and len(points_2d) >= 3:
                # Hacim hesapla
                volume_result = self.calculate_volume_from_points(points_2d, depth_img)
                if volume_result is not None:
                    print(f"Hesaplanan hacim: {volume_result:.6f} m³")
                    print(f"Hesaplanan hacim: {volume_result*1000:.2f} litre")
                else:
                    print("Hacim hesaplanamadı!")
            elif key == ord('r'):
                points_2d.clear()
                volume_result = None
                print("Sıfırlandı")
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def calculate_volume_from_points(self, points_2d, depth_img):
        """Seçilen noktalardan hacim hesapla"""
        if depth_img is None or len(points_2d) < 3:
            return None
        
        try:
            # 2D noktaları 3D'ye çevir
            points_3d = []
            
            for x, y in points_2d:
                if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
                    z = depth_img[y, x]
                    if not np.isnan(z) and not np.isinf(z) and z > 0:
                        points_3d.append([x, y, z])
            
            if len(points_3d) < 3:
                print("Yeterli geçerli 3D nokta yok!")
                return None
            
            points_3d = np.array(points_3d)
            
            # Basit hacim hesaplama yöntemleri
            volume = self.calculate_volume_methods(points_3d, points_2d, depth_img)
            
            return volume
            
        except Exception as e:
            print(f"Hacim hesaplama hatası: {e}")
            return None
    
    def calculate_volume_methods(self, points_3d, points_2d, depth_img):
        """Farklı hacim hesaplama yöntemleri"""
        
        # Yöntem 1: Çokgen tabanlı hacim (basit)
        volume_polygon = self.volume_by_polygon(points_2d, depth_img)
        
        # Yöntem 2: Delaunay üçgenleme ile hacim
        volume_delaunay = self.volume_by_delaunay(points_3d)
        
        # Yöntem 3: Konveks gövde hacmi
        volume_convex = self.volume_by_convex_hull(points_3d)
        
        print(f"\nFarklı yöntemlerle hesaplanan hacimler:")
        print(f"Çokgen tabanlı: {volume_polygon:.6f} m³")
        print(f"Delaunay: {volume_delaunay:.6f} m³") 
        print(f"Konveks gövde: {volume_convex:.6f} m³")
        
        # Ortalama değer döndür
        volumes = [v for v in [volume_polygon, volume_delaunay, volume_convex] if v is not None and v > 0]
        if volumes:
            return np.mean(volumes)
        
        return None
    
    def volume_by_polygon(self, points_2d, depth_img):
        """Çokgen alanı ve ortalama derinlik ile hacim"""
        try:
            # 2D çokgen alanını hesapla
            points = np.array(points_2d)
            area_2d = cv2.contourArea(points)
            
            # Çokgen içindeki ortalama derinliği hesapla
            mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            
            depths = depth_img[mask > 0]
            valid_depths = depths[~np.isnan(depths) & ~np.isinf(depths) & (depths > 0)]
            
            if len(valid_depths) == 0:
                return None
            
            avg_depth = np.mean(valid_depths)
            
            # Kamera kalibrasyonu (piksel -> metre dönüşümü)
            # Bu değerler kameranıza göre ayarlanmalı
            pixel_to_meter = avg_depth / 1000.0  # Basit yaklaşım
            
            area_3d = area_2d * (pixel_to_meter ** 2)
            volume = area_3d * avg_depth
            
            return volume
            
        except Exception as e:
            print(f"Çokgen hacim hesaplama hatası: {e}")
            return None
    
    def volume_by_delaunay(self, points_3d):
        """Delaunay üçgenleme ile hacim"""
        try:
            from scipy.spatial import Delaunay
            
            if len(points_3d) < 4:
                return None
            
            # 2D Delaunay üçgenleme (x, y koordinatları)
            points_2d = points_3d[:, :2]
            tri = Delaunay(points_2d)
            
            total_volume = 0
            
            for simplex in tri.simplices:
                # Üçgenin köşeleri
                p1, p2, p3 = points_3d[simplex]
                
                # Üçgen alanı
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * np.linalg.norm(np.cross(v1[:2], v2[:2]))
                
                # Ortalama yükseklik
                avg_height = np.mean([p1[2], p2[2], p3[2]])
                
                # Hacim katkısı
                total_volume += area * avg_height
            
            return total_volume
            
        except ImportError:
            print("scipy kütüphanesi gerekli: pip install scipy")
            return None
        except Exception as e:
            print(f"Delaunay hacim hesaplama hatası: {e}")
            return None
    
    def volume_by_convex_hull(self, points_3d):
        """Konveks gövde hacmi"""
        try:
            from scipy.spatial import ConvexHull
            
            if len(points_3d) < 4:
                return None
            
            hull = ConvexHull(points_3d)
            return hull.volume
            
        except ImportError:
            print("scipy kütüphanesi gerekli: pip install scipy")
            return None
        except Exception as e:
            print(f"Konveks gövde hacim hesaplama hatası: {e}")
            return None
    
    def object_volume_measurement(self):
        """Nesne segmentasyonu ile otomatik hacim ölçümü"""
        print("=== NESNE HACMİ ÖLÇÜMÜ ===")
        print("Talimatlar:")
        print("- 's': Segmentasyon yap ve hacim hesapla")
        print("- Threshold ayarları için trackbar kullanın")
        print("- 'q': Çıkış")
        
        # Trackbar oluştur
        cv2.namedWindow('Nesne Hacmi')
        cv2.createTrackbar('Min Depth (cm)', 'Nesne Hacmi', 30, 200, lambda x: None)
        cv2.createTrackbar('Max Depth (cm)', 'Nesne Hacmi', 150, 500, lambda x: None)
        cv2.createTrackbar('Min Area', 'Nesne Hacmi', 1000, 10000, lambda x: None)
        
        while True:
            left_img, _, depth_img = self.capture_frame()
            
            if left_img is not None and depth_img is not None:
                display_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                
                # Trackbar değerlerini al
                min_depth = cv2.getTrackbarPos('Min Depth (cm)', 'Nesne Hacmi') / 100.0
                max_depth = cv2.getTrackbarPos('Max Depth (cm)', 'Nesne Hacmi') / 100.0
                min_area = cv2.getTrackbarPos('Min Area', 'Nesne Hacmi')
                
                # Derinlik filtreleme
                depth_mask = ((depth_img >= min_depth) & 
                             (depth_img <= max_depth) & 
                             ~np.isnan(depth_img) & 
                             ~np.isinf(depth_img))
                
                # Morfolojik işlemler
                kernel = np.ones((5,5), np.uint8)
                depth_mask = depth_mask.astype(np.uint8) * 255
                depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
                depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
                
                # Konturları bul
                contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                total_volume = 0
                object_count = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    if area > min_area:
                        # Nesne maskesi oluştur
                        object_mask = np.zeros_like(depth_mask)
                        cv2.fillPoly(object_mask, [contour], 255)
                        
                        # Nesne hacmini hesapla
                        object_volume = self.calculate_object_volume(depth_img, object_mask)
                        
                        if object_volume > 0:
                            total_volume += object_volume
                            object_count += 1
                            
                            # Konturu ve bilgileri çiz
                            cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)
                            
                            # Nesne merkezi
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                cv2.putText(display_img, f"{object_volume*1000:.1f}L", 
                                          (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Toplam bilgiyi göster
                cv2.putText(display_img, f"Toplam: {total_volume*1000:.2f}L ({object_count} nesne)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Derinlik maskesini göster
                depth_colored = cv2.applyColorMap(depth_mask, cv2.COLORMAP_JET)
                combined = np.hstack((display_img, depth_colored))
                
                cv2.imshow('Nesne Hacmi', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def calculate_object_volume(self, depth_img, mask):
        """Maskelenmiş nesnenin hacmini hesapla"""
        try:
            # Maske içindeki derinlik değerleri
            masked_depth = depth_img * (mask / 255.0)
            valid_depths = masked_depth[masked_depth > 0]
            
            if len(valid_depths) == 0:
                return 0
            
            # Piksel alanı
            pixel_count = np.sum(mask > 0)
            
            # Kamera parametreleri (yaklaşık değerler)
            # Bu değerler kameranıza göre kalibre edilmelidir
            avg_depth = np.mean(valid_depths)
            
            # Piksel başına alan (m²)
            # FOV ve çözünürlük bilgisine göre hesaplanmalı
            pixel_area = (avg_depth * 0.001) ** 2  # Yaklaşık değer
            
            # Nesne taban alanı
            base_area = pixel_count * pixel_area
            
            # Ortalama yükseklik (zemin düzleminden)
            # Bu basit bir yaklaşım, daha doğrusu için zemin tespiti gerekir
            min_depth = np.min(valid_depths)
            avg_height = avg_depth - min_depth
            
            volume = base_area * avg_height
            return max(0, volume)
            
        except Exception as e:
            print(f"Nesne hacim hesaplama hatası: {e}")
            return 0
    
    def get_point_cloud(self):
        """3D nokta bulutu al"""
        if not self.is_opened:
            return None
        
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ)
        return self.point_cloud.get_data()
    
    def measure_distance(self, x, y):
        """Belirtilen pikseldeki mesafeyi ölç"""
        left_img, _, depth_img = self.capture_frame()
        
        if depth_img is not None and 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
            distance = depth_img[y, x]
            if not np.isnan(distance) and not np.isinf(distance):
                return distance
        
        return None
    
    def object_detection_distance(self):
        """Nesne tespiti ile mesafe ölçümü (basit örnek)"""
        print("Nesne tespiti ve mesafe ölçümü başladı. 'q' ile çıkın.")
        
        # Basit renk tabanlı nesne tespiti için HSV aralıkları
        # Mavi nesneler için örnek
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        while True:
            left_img, _, depth_img = self.capture_frame()
            
            if left_img is not None:
                # BGR'ye çevir ve HSV'ye dönüştür
                bgr_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
                
                # Mavi nesneleri tespit et
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Minimum alan
                        # Nesnenin merkez noktasını bul
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Mesafe ölç
                            distance = self.measure_distance(cx, cy)
                            
                            # Nesneyi çiz
                            cv2.drawContours(bgr_img, [contour], -1, (0, 255, 0), 2)
                            cv2.circle(bgr_img, (cx, cy), 5, (255, 0, 0), -1)
                            
                            if distance is not None:
                                cv2.putText(bgr_img, f"Mesafe: {distance:.2f}m", 
                                          (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Nesne Tespiti ve Mesafe', bgr_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def record_video(self, duration=10, filename=None):
        """Video kaydet"""
        if not self.is_opened:
            print("Önce kamerayı açın!")
            return
        
        if filename is None:
            filename = f"zed_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        
        # Video writer ayarları
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30
        
        # İlk frame'i al boyut için
        left_img, _, _ = self.capture_frame()
        if left_img is None:
            print("Frame alınamadı!")
            return
        
        bgr_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
        height, width = bgr_img.shape[:2]
        
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        print(f"{duration} saniye video kaydediliyor...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            left_img, _, _ = self.capture_frame()
            
            if left_img is not None:
                bgr_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                out.write(bgr_img)
                
                # İlerleme göster
                elapsed = time.time() - start_time
                print(f"\rKayıt: {elapsed:.1f}/{duration}s", end="", flush=True)
        
        out.release()
        print(f"\nVideo kaydedildi: {filename}")
    
    def close(self):
        """Kamerayı kapat"""
        if self.is_opened:
            self.zed.close()
            self.is_opened = False
            print("ZED kamera kapatıldı")


# Kullanım örneği ve ana program
def main():
    """Ana program"""
    print("=== ZED KAMERA KONTROLCÜSÜ ===")
    
    # ZED SDK kontrolü
    try:
        import pyzed.sl as sl
    except ImportError:
        print("HATA: pyzed kütüphanesi bulunamadı!")
        print("Kurulum: pip install pyzed")
        return
    
    zed_cam = ZEDCamera()
    
    try:
        # Kamerayı aç
        print("Kamera bağlantısı kontrol ediliyor...")
        if not zed_cam.open_camera():
            print("\nKamera açılamadı. Programdan çıkılıyor...")
            return
        
        print("\n=== ZED KAMERA MENÜSÜ ===")
        print("1. Canlı görüntü (sadece sol)")
        print("2. Canlı görüntü + derinlik")
        print("3. Video kaydet")
        print("4. Nesne tespiti ve mesafe ölçümü")
        print("5. Görüntü yakala ve kaydet")
        print("6. Manuel hacim ölçümü")
        print("7. Otomatik nesne hacmi ölçümü")
        print("8. Kamera bilgilerini göster")
        print("9. Çıkış")
        
        while True:
            try:
                secim = input("\nSeçiminizi yapın (1-9): ").strip()
                
                if secim == "1":
                    print("Canlı görüntü başlatılıyor...")
                    zed_cam.live_stream(show_depth=False)
                elif secim == "2":
                    print("Canlı görüntü + derinlik başlatılıyor...")
                    zed_cam.live_stream(show_depth=True)
                elif secim == "3":
                    try:
                        sure = int(input("Kayıt süresi (saniye): "))
                        if sure > 0:
                            zed_cam.record_video(duration=sure)
                        else:
                            print("Süre pozitif olmalı!")
                    except ValueError:
                        print("Geçerli bir sayı girin!")
                elif secim == "4":
                    print("Nesne tespiti başlatılıyor...")
                    zed_cam.object_detection_distance()
                elif secim == "5":
                    print("Görüntü yakalanıyor...")
                    left_img, right_img, depth_img = zed_cam.capture_frame()
                    if left_img is not None:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        left_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                        cv2.imwrite(f'zed_left_{timestamp}.jpg', left_bgr)
                        
                        if depth_img is not None:
                            depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                            cv2.imwrite(f'zed_depth_{timestamp}.jpg', depth_normalized)
                        
                        print(f"Görüntüler kaydedildi: zed_left_{timestamp}.jpg")
                    else:
                        print("Görüntü yakalanamadı!")
                elif secim == "6":
                    print("Manuel hacim ölçümü başlatılıyor...")
                    zed_cam.volume_measurement()
                elif secim == "7":
                    print("Otomatik nesne hacmi ölçümü başlatılıyor...")
                    zed_cam.object_volume_measurement()
                elif secim == "8":
                    zed_cam.show_camera_info()
                elif secim == "9":
                    print("Çıkılıyor...")
                    break
                else:
                    print("Geçersiz seçim! (1-9 arası)")
                    
            except ValueError:
                print("Lütfen geçerli bir sayı girin!")
            except KeyboardInterrupt:
                print("\nProgram durduruldu.")
                break
            except Exception as e:
                print(f"Hata oluştu: {e}")
                print("Program devam ediyor...")
    
    except Exception as e:
        print(f"Kritik hata: {e}")
    finally:
        zed_cam.close()
        print("Program sonlandırıldı.")


# Kurulum talimatları
def installation_guide():
    """Kurulum rehberi"""
    guide = """
    === ZED SDK KURULUM REHBERİ ===
    
    1. ZED SDK İndir:
       - https://www.stereolabs.com/developers/release/ adresinden
       - İşletim sisteminize uygun sürümü indirin
    
    2. Python API Kurulumu:
       pip install pyzed
    
    3. Gerekli kütüphaneler:
       pip install opencv-python numpy
    
    4. CUDA Kurulumu (önerilen):
       - NVIDIA GPU varsa CUDA toolkit kurun
       - Daha hızlı işlem için gerekli
    
    5. Kamera Bağlantısı:
       - ZED kamerayı USB 3.0 portuna bağlayın
       - Yeterli güç kaynağı olduğundan emin olun
    
    === KULLANIM İPUÇLARI ===
    
    - İyi aydınlatma önemlidir
    - Kamerayı sabit tutun (tripod önerilir)
    - Derinlik ölçümü için yeterli doku gerekli
    - Çok yakın nesneler (< 0.3m) ölçülemeyebilir
    """
    print(guide)


if __name__ == "__main__":
    print("ZED Kamera Kontrolcüsü")
    print("Kurulum rehberi için 'installation_guide()' çalıştırın")
    main()