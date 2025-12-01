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
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.2
        
        self.image_left = sl.Mat()
        self.depth_map = sl.Mat()
        self.is_opened = False
    
    def open_camera(self):
        print("ZED kamera açılıyor...")
        if not self.check_camera_connection(): return False
        
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Kamera açma hatası: {err}")
            return False
        
        self.is_opened = True
        print("ZED kamera başarıyla açıldı.")
        self.show_camera_info()
        return True

    def check_camera_connection(self):
        cameras = sl.Camera.get_device_list()
        if len(cameras) == 0:
            print("HATA: ZED kamera bulunamadı!")
            return False
        print(f"Bulunan kameralar: {len(cameras)}")
        return True

    def show_camera_info(self):
        """Kamera bilgilerini göster (SON DÜZELTİLMİŞ VERSİYON)"""
        if not self.is_opened: return
        
        # Bu nesne, kameranın o anki GERÇEK çalışan parametrelerini tutar.
        cam_info = self.zed.get_camera_information()
        
        print(f"\nKamera Bilgileri:")
        print(f"  Seri No: {cam_info.serial_number}")
        
        if hasattr(cam_info, 'camera_model'):
            print(f"  Model: {cam_info.camera_model}")

        # --- DÜZELTME BAŞLANGICI ---
        # Aktif çözünürlüğü almanın doğru ve en güvenilir yolu,
        # CameraInformation nesnesinin içindeki camera_configuration'dan okumaktır.
        resolution = cam_info.camera_configuration.resolution
        print(f"  Çalışan Çözünürlük: {resolution.width}x{resolution.height}")
        # --- DÜZELTME SONU ---
        
        # İstenen FPS ve Derinlik modunu teyit için yine init_params'tan okuyabiliriz.
        init_params = self.zed.get_init_parameters()
        print(f"  Çalışan FPS: {cam_info.camera_configuration.fps}")
        print(f"  Derinlik Modu: {init_params.depth_mode}")

    def capture_frame(self):
        if not self.is_opened:
            return None, None
        
        runtime_params = sl.RuntimeParameters()
        if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            return self.image_left.get_data(), self.depth_map.get_data()
        
        return None, None

    def show_center_distance(self):
        """
        Canlı görüntüde ekranın tam ortasındaki mesafeyi ölçer ve gösterir.
        """
        print("\nMerkez nokta mesafesi gösteriliyor. 'q' ile çıkın.")
        print("İpucu: Kamerayı farklı mesafelerdeki nesnelere doğrultun.")
        
        while True:
            left_img, depth_img = self.capture_frame()
            
            if left_img is not None and depth_img is not None:
                h, w, _ = left_img.shape
                center_x, center_y = w // 2, h // 2
                
                distance = depth_img[center_y, center_x]
                
                display_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                
                text = ""
                marker_color = (0, 0, 255) # Kırmızı (Ölçülemiyor)
                if not np.isnan(distance) and not np.isinf(distance) and distance > 0:
                    text = f"Mesafe: {distance:.2f} metre"
                    marker_color = (0, 255, 0) # Yeşil (Ölçüldü)
                else:
                    text = "Mesafe: Olculemiyor"
                    
                # Artı işaretini çiz
                cv2.line(display_img, (center_x - 10, center_y), (center_x + 10, center_y), marker_color, 2)
                cv2.line(display_img, (center_x, center_y - 10), (center_x, center_y + 10), marker_color, 2)
                
                # Metni yazdır
                cv2.putText(display_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow("Merkez Nokta Mesafe Ölçümü", display_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def close(self):
        if self.is_opened:
            self.zed.close()
            self.is_opened = False
            print("ZED kamera kapatıldı")

def main():
    print("=== ZED KAMERA KONTROLCÜSÜ ===")
    
    zed_cam = ZEDCamera()
    
    try:
        if not zed_cam.open_camera():
            print("\nKamera açılamadı. Programdan çıkılıyor...")
            return
        
        menu = """
=== ZED KAMERA MENÜSÜ ===
1. Merkez Nokta Mesafesini Göster (En Basit Test)
2. Görüntü yakala ve kaydet
3. Kamera bilgilerini göster
4. Çıkış
"""
        print(menu)
        
        while True:
            secim = input("\nSeçiminizi yapın (1-4): ").strip()
            
            if secim == "1":
                zed_cam.show_center_distance()
            elif secim == "2":
                # ... (Bu kısım aynı)
                pass
            elif secim == "3":
                zed_cam.show_camera_info()
            elif secim == "4":
                print("Çıkılıyor...")
                break
            else:
                print("Geçersiz seçim! (1-4 arası)")
                    
    except Exception as e:
        print(f"Kritik hata: {e}")
    finally:
        zed_cam.close()
        print("Program sonlandırıldı.")

if __name__ == "__main__":
    main()