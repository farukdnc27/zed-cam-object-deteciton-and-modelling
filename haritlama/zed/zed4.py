import pyzed.sl as sl
import cv2
import numpy as np
import time
from datetime import datetime
import os

class ZEDCamera:
    """
    ZED Kamera için temel ve gelişmiş işlemler sınıfı.
    Hacim hesaplama mantığı, yüzey tespiti ile iyileştirilmiştir.
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

    def open_camera(self):
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
        cameras = sl.Camera.get_device_list()
        if not cameras:
            print("\nHATA: ZED kamera bulunamadı!")
            return False
        print(f"Bulunan ZED kameralar: {len(cameras)}")
        return True
    
    def _fetch_camera_parameters(self):
        if self.is_opened:
            cam_info = self.zed.get_camera_information()
            self.camera_intrinsics = cam_info.camera_configuration.calibration_parameters.left_cam
            print("\nKamera içsel parametreleri (intrinsics) yüklendi.")

    def _handle_camera_error(self, error_code):
        print(f"\nKamera Hatası: {repr(error_code)}")

    def show_camera_info(self):
        if not self.is_opened: return
        try:
            cam_info = self.zed.get_camera_information()
            print("\n--- Kamera Bilgileri ---")
            print(f"Model: {cam_info.camera_model}")
            print(f"Seri No: {cam_info.serial_number}")
            print(f"Firmware: {cam_info.camera_firmware_version}")
            print(f"Çözünürlük: {int(cam_info.camera_resolution.width)}x{int(cam_info.camera_resolution.height)}")
            print(f"FPS: {self.zed.get_init_parameters().camera_fps}")
            print("------------------------")
        except Exception as e:
            print(f"Kamera bilgileri alınırken hata: {e}")

    def capture_frame(self):
        if not self.is_opened: return None, None
        
        runtime_params = sl.RuntimeParameters()
        if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            return self.image_left.get_data(), self.depth_map.get_data()
        
        return None, None

    def live_stream(self, show_depth=True):
        if not self.is_opened:
            print("Önce kamerayı açın!"); return
        
        print("\nCanlı akış başladı. Çıkmak için 'q' tuşuna basın.")
        
        try:
            while True:
                left_img, depth_img = self.capture_frame()
                if left_img is not None:
                    display_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                    
                    if show_depth and depth_img is not None:
                        depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        combined = np.hstack((display_img, depth_colored))
                        cv2.imshow('ZED | Sol Görüntü & Derinlik Haritası', combined)
                    else:
                        cv2.imshow('ZED | Sol Görüntü', display_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            cv2.destroyAllWindows()
            
    def object_volume_measurement(self):
        print("\n=== OTOMATİK NESNE HACMİ ÖLÇÜMÜ (Akıllı Yöntem) ===")
        print("Talimatlar:")
        print("- Yöntem artık nesnenin durduğu yüzeyi bularak yüksekliği hesaplar.")
        print("- 'q': Menüye geri dönün.")

        window_name = 'Otomatik Nesne Hacmi (Akıllı Yöntem)'
        cv2.namedWindow(window_name)
        cv2.createTrackbar('Min Depth (cm)', window_name, 30, 300, lambda x: None)
        cv2.createTrackbar('Max Depth (cm)', window_name, 150, 500, lambda x: None)
        cv2.createTrackbar('Min Alan (px)', window_name, 1000, 50000, lambda x: None)

        try:
            while True:
                left_img, depth_img = self.capture_frame()
                if left_img is None or depth_img is None: continue

                display_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                
                min_depth = cv2.getTrackbarPos('Min Depth (cm)', window_name) / 100.0
                max_depth = cv2.getTrackbarPos('Max Depth (cm)', window_name) / 100.0
                min_area = cv2.getTrackbarPos('Min Alan (px)', window_name)

                mask = cv2.inRange(depth_img, min_depth, max_depth)
                
                kernel = np.ones((7, 7), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                total_volume_L = 0
                for contour in contours:
                    if cv2.contourArea(contour) > min_area:
                        # YENİ ve AKILLI HACİM HESAPLAMA FONKSİYONU
                        volume_m3 = self._calculate_volume_from_bounding_box(depth_img, contour)
                        
                        if volume_m3 is not None and volume_m3 > 0:
                            total_volume_L += volume_m3 * 1000
                            
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(display_img, f"{volume_m3*1000:.2f} L", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.putText(display_img, f"Toplam Hacim: {total_volume_L:.2f} L", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                mask_colored = cv2.applyColorMap(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_VIRIDIS)
                combined = np.hstack((display_img, mask_colored))
                cv2.imshow(window_name, combined)

                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            cv2.destroyAllWindows()

    def _calculate_volume_from_bounding_box(self, depth_map, contour):
        """
        (YENİ AKILLI YÖNTEM)
        Nesnenin üzerinde durduğu yüzeyi bularak hacmini tahmin eder.
        """
        if self.camera_intrinsics is None: return None

        # 1. Gerekli maskeleri ve kontur bilgilerini al
        object_mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(contour)

        # 2. Nesnenin tepe noktalarının ortalama mesafesini bul
        object_pixels_depth = depth_map[object_mask > 0]
        valid_object_depths = object_pixels_depth[~np.isnan(object_pixels_depth) & ~np.isinf(object_pixels_depth)]
        if valid_object_depths.size == 0:
            return 0.0
        # Medyan, aykırı değerlere karşı daha robusttur
        object_top_z = np.median(valid_object_depths)

        # 3. Nesnenin durduğu yüzeyin mesafesini bul (En kritik adım)
        #    Konturun etrafında bir "halka" oluşturup oranın mesafesini alıyoruz.
        ring_kernel = np.ones((25, 25), np.uint8) # Halkanın kalınlığı
        dilated_mask = cv2.dilate(object_mask, ring_kernel, iterations=1)
        surface_mask = dilated_mask - object_mask
        
        surface_pixels_depth = depth_map[surface_mask > 0]
        valid_surface_depths = surface_pixels_depth[~np.isnan(surface_pixels_depth) & ~np.isinf(surface_pixels_depth)]
        
        if valid_surface_depths.size < 20: # Yeterli yüzey noktası yoksa, eski yönteme (arkadaki duvara) geri dön
            surface_z = np.max(valid_object_depths)
        else:
            surface_z = np.median(valid_surface_depths)

        # 4. Ortalama Yüksekliği Hesapla (Yüzeye göre)
        avg_height_m = surface_z - object_top_z
        # Yükseklik negatifse veya çok küçükse (düz bir yüzey gibi), hacim sıfırdır.
        if avg_height_m <= 0.01: 
            return 0.0

        # 5. Taban Alanını Hesapla (Gerçek Dünya Değerleri)
        fx = self.camera_intrinsics.fx
        fy = self.camera_intrinsics.fy
        
        # Alanı, nesnenin kendi mesafesine göre ölçeklendiriyoruz
        real_width_m = (w * object_top_z) / fx
        real_height_m = (h * object_top_z) / fy # Görüntüdeki 'h' değeri
        base_area_m2 = real_width_m * real_height_m

        # 6. Nihai Hacmi Hesapla
        volume_m3 = base_area_m2 * avg_height_m
        
        return volume_m3

    def close(self):
        if self.is_opened:
            self.zed.close()
            self.is_opened = False
            print("\nZED kamera kapatıldı.")

# --- Ana Program ---
def main():
    print("=== ZED KAMERA KONTROL MERKEZİ ===")
    zed_cam = ZEDCamera(resolution=sl.RESOLUTION.HD720, fps=30)
    
    try:
        if not zed_cam.open_camera():
            print("\nKamera açılamadı. Program sonlandırılıyor."); return

        # Menüyü basitleştirelim, şimdilik sadece ilgili fonksiyonlar
        menu = {
            "1": "Canlı Görüntü + Derinlik",
            "2": "Otomatik Nesne Hacmi Ölçümü (Akıllı Yöntem)",
            "3": "Çıkış"
        }

        while True:
            print("\n--- ANA MENÜ ---")
            for key, value in menu.items(): print(f"{key}. {value}")
            
            choice = input("Seçiminizi yapın: ").strip()
            
            if choice == "1":
                zed_cam.live_stream(show_depth=True)
            elif choice == "2":
                zed_cam.object_volume_measurement()
            elif choice == "3":
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