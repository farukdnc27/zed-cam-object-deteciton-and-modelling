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
    YOLOv8 ile tespit edilen nesnenin türüne göre (Silindir/Kutu)
    özel hacim hesaplama modelleri kullanır.
    """

    def __init__(self, resolution=sl.RESOLUTION.HD720, fps=30):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Daha hızlı ve daha az gürültülü olabilir
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
                print("YOLOv8 modeli yükleniyor...")
                self.yolo_model = YOLO('yolov8n.pt')
                print("YOLOv8 modeli başarıyla yüklendi.")
            except Exception as e:
                print(f"HATA: YOLO modeli yüklenemedi: {e}")
                self.yolo_model = None
        else:
            print("UYARI: 'ultralytics' kütüphanesi bulunamadı. Kurulum: pip install ultralytics")

    def open_camera(self):
        print("ZED kamera açılıyor...")
        if not self._check_camera_connection(): return False
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self._handle_camera_error(err); return False
        self.is_opened = True
        self._fetch_camera_parameters()
        print("ZED kamera başarıyla açıldı.")
        return True

    def _check_camera_connection(self):
        cameras = sl.Camera.get_device_list()
        if not cameras: print("\nHATA: ZED kamera bulunamadı!"); return False
        return True
    
    def _fetch_camera_parameters(self):
        if self.is_opened:
            self.camera_intrinsics = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
            print("Kamera içsel parametreleri (intrinsics) yüklendi.")

    def _handle_camera_error(self, error_code):
        print(f"\nKamera Hatası: {repr(error_code)}")

    def capture_frame(self):
        if not self.is_opened: return None, None
        runtime_params = sl.RuntimeParameters()
        if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            return self.image_left.get_data(), self.depth_map.get_data()
        return None, None

    def yolo_volume_measurement(self):
        if self.yolo_model is None:
            print("HATA: YOLO modeli kullanılamıyor."); return

        print("\n=== YOLO ile Akıllı Hacim Ölçümü ===")
        print("Tespit edilen nesnenin türüne göre Silindir/Kutu modeli kullanılacaktır.")
        print("Çıkmak için 'q' tuşuna basın.")
        
        window_name = "YOLO ile Akıllı Hacim Ölçümü"
        cv2.namedWindow(window_name)

        try:
            while True:
                left_img, depth_img = self.capture_frame()
                if left_img is None: continue

                bgr_img = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
                results = self.yolo_model(bgr_img, verbose=False, conf=0.5)

                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

                    volume_m3 = None
                    model_used = ""

                    # NESNE TÜRÜNE GÖRE MODEL SEÇİMİ
                    if class_name in ['bottle', 'cup', 'vase']:
                        volume_m3 = self._calculate_cylinder_volume(depth_img, contour)
                        model_used = "Silindir"
                    else: # Diğer tüm nesneler için Kutu modeli varsayalım
                        volume_m3 = self._calculate_cuboid_volume(depth_img, contour)
                        model_used = "Kutu"
                    
                    # Sonuçları ekrana çiz
                    cv2.rectangle(bgr_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{class_name} ({model_used})"
                    if volume_m3 is not None and volume_m3 > 0:
                        label_text += f": {volume_m3*1000:.2f} L"

                    cv2.putText(bgr_img, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow(window_name, bgr_img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            cv2.destroyAllWindows()

    def _calculate_cuboid_volume(self, depth_map, contour):
        """Kutu (Prizma) şeklindeki nesneler için hacim hesaplar."""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Yükseklik ve taban alanı hesaplamak için ortak mantık
        object_top_z, surface_z = self._get_object_and_surface_depth(depth_map, contour)
        if object_top_z is None or surface_z is None: return 0.0

        height_m = surface_z - object_top_z
        if height_m <= 0.01: return 0.0

        # Kutu modeli için taban alanı
        fx = self.camera_intrinsics.fx
        fy = self.camera_intrinsics.fy
        real_width_m = (w * object_top_z) / fx
        real_depth_m_approx = (h * object_top_z) / fy # Bu en zayıf varsayım
        base_area_m2 = real_width_m * real_depth_m_approx

        return base_area_m2 * height_m

    def _calculate_cylinder_volume(self, depth_map, contour):
        """Silindir şeklindeki nesneler için hacim hesaplar."""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Yükseklik hesaplamak için ortak mantık
        object_top_z, surface_z = self._get_object_and_surface_depth(depth_map, contour)
        if object_top_z is None or surface_z is None: return 0.0

        height_m = surface_z - object_top_z
        if height_m <= 0.01: return 0.0

        # Silindir modeli için taban alanı (Dairesel)
        fx = self.camera_intrinsics.fx
        diameter_m = (w * object_top_z) / fx # Bounding box genişliği çap olarak alınır
        radius_m = diameter_m / 2.0
        base_area_m2 = np.pi * (radius_m ** 2)

        return base_area_m2 * height_m

    def _get_object_and_surface_depth(self, depth_map, contour):
        """Bir nesnenin ve üzerinde durduğu yüzeyin derinliğini bulan yardımcı fonksiyon."""
        object_mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)

        object_pixels_depth = depth_map[object_mask > 0]
        valid_object_depths = object_pixels_depth[~np.isnan(object_pixels_depth) & ~np.isinf(object_pixels_depth)]
        if valid_object_depths.size < 20: return None, None
        object_top_z = np.median(valid_object_depths)

        ring_kernel = np.ones((30, 30), np.uint8)
        dilated_mask = cv2.dilate(object_mask, ring_kernel, iterations=1)
        surface_mask = dilated_mask - object_mask
        
        surface_pixels_depth = depth_map[surface_mask > 0]
        valid_surface_depths = surface_pixels_depth[~np.isnan(surface_pixels_depth) & ~np.isinf(surface_pixels_depth)]
        
        if valid_surface_depths.size < 50:
            return None, None # Güvenilir yüzey bulunamadı
        surface_z = np.median(valid_surface_depths)
        
        return object_top_z, surface_z

    def close(self):
        if self.is_opened:
            self.zed.close()
            print("\nZED kamera kapatıldı.")

# --- Ana Program ---
def main():
    print("=== ZED KAMERA KONTROL MERKEZİ (Akıllı Modeller) ===")
    zed_cam = ZEDCamera()
    
    try:
        if not zed_cam.open_camera():
            print("\nKamera açılamadı."); return

        zed_cam.yolo_volume_measurement()

    except Exception as e:
        print(f"\nBeklenmedik bir kritik hata oluştu: {e}")
    finally:
        zed_cam.close()
        print("Program sonlandırıldı.")

if __name__ == "__main__":
    main()