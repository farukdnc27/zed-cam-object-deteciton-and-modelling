import pyzed.sl as sl
import cv2
import numpy as np
import time
from datetime import datetime
import os

# YOLO entegrasyonu için try-except bloğu
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ZEDCamera:
    """
    ZED Kamera için temel ve gelişmiş işlemler sınıfı.
    YOLOv8 ile nesne tespiti ve akıllı hacim ölçümü entegre edilmiştir.
    """

    def __init__(self, resolution=sl.RESOLUTION.HD720, fps=30):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.3

        self.image_left = sl.Mat()
        self.depth_map = sl.Mat()
        
        self.is_opened = False
        self.camera_intrinsics = None
        
        # YOLO Modelini Yükle
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                print("YOLOv8 modeli yükleniyor (ilk çalıştırmada indirilebilir)...")
                # 'yolov8n.pt' -> nano (hızlı), 'yolov8s.pt' -> small (daha doğru)
                self.yolo_model = YOLO('yolov8n.pt') 
                print("YOLOv8 modeli başarıyla yüklendi.")
            except Exception as e:
                print(f"HATA: YOLO modeli yüklenemedi: {e}")
                self.yolo_model = None
        else:
            print("UYARI: 'ultralytics' kütüphanesi bulunamadı. YOLO fonksiyonları çalışmayacak.")
            print("Kurulum için: pip install ultralytics")


    def open_camera(self):
        # ... (Bu fonksiyon aynı kalıyor, değişiklik yok)
        print("ZED kamera açılıyor...")
        if not self._check_camera_connection():
            return False
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self._handle_camera_error(err)
            return False
        self.is_opened = True
        self._fetch_camera_parameters()
        print("ZED kamera başarıyla açıldı.")
        self.show_camera_info()
        return True

    def _check_camera_connection(self):
        # ... (Bu fonksiyon aynı kalıyor, değişiklik yok)
        cameras = sl.Camera.get_device_list()
        if not cameras: print("\nHATA: ZED kamera bulunamadı!"); return False
        print(f"Bulunan ZED kameralar: {len(cameras)}"); return True
    
    def _fetch_camera_parameters(self):
        # ... (Bu fonksiyon aynı kalıyor, değişiklik yok)
        if self.is_opened:
            cam_info = self.zed.get_camera_information()
            self.camera_intrinsics = cam_info.camera_configuration.calibration_parameters.left_cam
            print("\nKamera içsel parametreleri (intrinsics) yüklendi.")

    def _handle_camera_error(self, error_code):
        # ... (Bu fonksiyon aynı kalıyor, değişiklik yok)
        print(f"\nKamera Hatası: {repr(error_code)}")

    def show_camera_info(self):
        # ... (Bu fonksiyon aynı kalıyor, değişiklik yok)
        if not self.is_opened: return
        try:
            cam_info = self.zed.get_camera_information()
            print("\n--- Kamera Bilgileri ---")
            print(f"Model: {cam_info.camera_model}")
            print(f"Seri No: {cam_info.serial_number}")
            # ... (geri kalanı aynı)
        except Exception as e:
            print(f"Kamera bilgileri alınırken hata: {e}")

    def capture_frame(self):
        # ... (Bu fonksiyon aynı kalıyor, değişiklik yok)
        if not self.is_opened: return None, None
        runtime_params = sl.RuntimeParameters()
        if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            return self.image_left.get_data(), self.depth_map.get_data()
        return None, None

    def yolo_volume_measurement(self):
        """
        YOLOv8 ile nesneleri tespit eder ve her birinin hacmini hesaplar.
        """
        if self.yolo_model is None:
            print("HATA: YOLO modeli kullanılamıyor. Lütfen kurulumu kontrol edin.")
            return

        print("\n=== YOLO ile Nesne Tespiti ve Hacim Ölçümü ===")
        print("Tespit edilen her nesnenin hacmi hesaplanacaktır.")
        print("Çıkmak için 'q' tuşuna basın.")
        
        window_name = "YOLOv8 ile Hacim Ölçümü"
        cv2.namedWindow(window_name)

        try:
            while True:
                left_img, depth_img = self.capture_frame()
                if left_img is None or depth_img is None: continue

                # Görüntüyü OpenCV formatına çevir
                bgr_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)

                # YOLO ile nesne tespiti yap
                # verbose=False, konsola sürekli çıktı basmasını engeller
                results = self.yolo_model(bgr_img, verbose=False)

                # Tespit edilen her nesne için döngü
                for box in results[0].boxes:
                    # Düşük güvenilirlikteki tespitleri filtrele
                    if box.conf[0] > 0.45:
                        # Sınırlayıcı kutu koordinatlarını al (x1, y1, x2, y2)
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        
                        # Sınırlayıcı kutuyu bir kontura çevirerek hacim fonksiyonuna gönder
                        contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                        
                        volume_m3 = self._calculate_volume_with_surface_detection(depth_img, contour)
                        
                        # Sonuçları ekrana çiz
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        conf_score = float(box.conf[0])
                        
                        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label_text = f"{class_name} ({conf_score:.2f})"
                        if volume_m3 is not None and volume_m3 > 0:
                            label_text += f": {volume_m3*1000:.2f} L"

                        cv2.putText(bgr_img, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow(window_name, bgr_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()

    def _calculate_volume_with_surface_detection(self, depth_map, contour):
        """
        (YENİ AKILLI YÖNTEM)
        Nesnenin üzerinde durduğu yüzeyi bularak hacmini tahmin eder.
        Bu fonksiyon hem manuel çizim hem de YOLO kutusu ile çalışabilir.
        """
        if self.camera_intrinsics is None: return None

        object_mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(contour)

        object_pixels_depth = depth_map[object_mask > 0]
        valid_object_depths = object_pixels_depth[~np.isnan(object_pixels_depth) & ~np.isinf(object_pixels_depth)]
        if valid_object_depths.size < 10: return 0.0
        object_top_z = np.median(valid_object_depths)

        ring_kernel = np.ones((25, 25), np.uint8)
        dilated_mask = cv2.dilate(object_mask, ring_kernel, iterations=1)
        surface_mask = dilated_mask - object_mask
        
        surface_pixels_depth = depth_map[surface_mask > 0]
        valid_surface_depths = surface_pixels_depth[~np.isnan(surface_pixels_depth) & ~np.isinf(surface_pixels_depth)]
        
        if valid_surface_depths.size < 20:
            surface_z = np.max(valid_object_depths)
        else:
            surface_z = np.median(valid_surface_depths)

        avg_height_m = surface_z - object_top_z
        if avg_height_m <= 0.01: return 0.0

        fx, fy = self.camera_intrinsics.fx, self.camera_intrinsics.fy
        real_width_m = (w * object_top_z) / fx
        real_height_m = (h * object_top_z) / fy
        base_area_m2 = real_width_m * real_height_m

        volume_m3 = base_area_m2 * avg_height_m
        
        return volume_m3

    def close(self):
        if self.is_opened:
            self.zed.close()
            self.is_opened = False
            print("\nZED kamera kapatıldı.")


# --- Ana Program ---
def main():
    print("=== ZED KAMERA KONTROL MERKEZİ (YOLO Entegreli) ===")
    zed_cam = ZEDCamera(resolution=sl.RESOLUTION.HD720, fps=30)
    
    try:
        if not zed_cam.open_camera():
            print("\nKamera açılamadı. Program sonlandırılıyor."); return

        menu = {
            "1": "YOLO ile Nesne Tespiti ve Hacim Ölçümü",
            "2": "Çıkış"
        }

        while True:
            print("\n--- ANA MENÜ ---")
            for key, value in menu.items(): print(f"{key}. {value}")
            
            choice = input("Seçiminizi yapın: ").strip()
            
            if choice == "1":
                zed_cam.yolo_volume_measurement()
            elif choice == "2":
                break
            else:
                print("Geçersiz seçim.")

    except Exception as e:
        print(f"\nBeklenmedik bir kritik hata oluştu: {e}")
    finally:
        zed_cam.close()
        print("Program sonlandırıldı.")

if __name__ == "__main__":
    main()