import pyzed.sl as sl
import cv2
import numpy as np
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ZEDCamera:
    """
    ZED Kamera için nihai versiyon. YOLOv8 ile tespit edilen nesnenin 
    türüne göre (Silindir/Kutu) ve Bounding Box boyutlarına dayalı
    sağlam bir hacim hesaplama modeli kullanır.
    """

    def __init__(self, resolution=sl.RESOLUTION.HD720, fps=30):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.3

        self.image_left = sl.Mat()
        self.depth_map = sl.Mat()
        
        self.is_opened = False
        self.camera_intrinsics = None
        
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
        if not sl.Camera.get_device_list():
            print("\nHATA: ZED kamera bulunamadı!"); return False
        
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"\nKamera Hatası: {repr(err)}"); return False
        
        self.is_opened = True
        self.camera_intrinsics = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        print("Kamera başarıyla açıldı ve içsel parametreler yüklendi.")
        return True

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

        print("\n=== YOLO ile Akıllı Hacim Ölçümü (Nihai Yöntem) ===")
        print("Yükseklik ve genişlik doğrudan Bounding Box'tan hesaplanır.")
        print("Çıkmak için 'q' tuşuna basın.")
        
        window_name = "YOLO ile Akıllı Hacim Ölçümü (Nihai Yöntem)"
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
                    
                    # YOLO kutusundan model tipini belirle
                    model_type = 'cylinder' if class_name in ['bottle', 'cup', 'vase'] else 'cuboid'
                    
                    # Tek ve birleşik hacim hesaplama fonksiyonunu çağır
                    volume_m3 = self._calculate_volume_from_bbox(depth_img, box, model_type)
                    
                    # Sonuçları ekrana çiz
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(bgr_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{class_name} ({model_type.capitalize()})"
                    if volume_m3 is not None and volume_m3 > 0:
                        label_text += f": {volume_m3*1000:.2f} L"

                    cv2.putText(bgr_img, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow(window_name, bgr_img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            cv2.destroyAllWindows()

    def _calculate_volume_from_bbox(self, depth_map, box, model_type):
        """
        YOLO'nun sınırlayıcı kutusunu (bounding box) kullanarak,
        belirtilen modele göre (silindir/kutu) hacim hesaplar.
        """
        if self.camera_intrinsics is None: return 0.0

        # 1. Bounding box koordinatlarını ve boyutlarını al
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        w_pixels = x2 - x1
        h_pixels = y2 - y1

        # 2. Nesnenin ortalama mesafesini (z) bul
        #    Sadece kutunun içindeki pikselleri dikkate al
        object_mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
        object_mask[y1:y2, x1:x2] = 255
        
        object_depths = depth_map[object_mask > 0]
        valid_depths = object_depths[~np.isnan(object_depths) & ~np.isinf(object_depths)]
        if valid_depths.size < 50: return 0.0 # Güvenilir ölçüm için yeterli nokta yok
        
        object_z = np.median(valid_depths)

        # 3. Piksel boyutlarını gerçek dünya boyutlarına çevir
        fx = self.camera_intrinsics.fx
        fy = self.camera_intrinsics.fy

        # YÜKSEKLİK: Bounding box'ın piksel yüksekliğinden hesaplanır. Bu en sağlam yöntem.
        height_m = (h_pixels * object_z) / fy
        
        # GENİŞLİK: Bounding box'ın piksel genişliğinden hesaplanır.
        width_m = (w_pixels * object_z) / fx

        # 4. Modele göre taban alanını ve hacmi hesapla
        base_area_m2 = 0
        if model_type == 'cylinder':
            # Genişliği çap olarak kabul et
            radius_m = width_m / 2.0
            base_area_m2 = np.pi * (radius_m ** 2)
        
        elif model_type == 'cuboid':
            # Derinliği göremediğimiz için en makul varsayım tabanın kare olmasıdır.
            # Dolayısıyla derinlik = genişlik
            depth_m = width_m 
            base_area_m2 = width_m * depth_m

        if base_area_m2 == 0: return 0.0

        volume_m3 = base_area_m2 * height_m
        return volume_m3

    def close(self):
        if self.is_opened:
            self.zed.close()
            print("\nZED kamera kapatıldı.")

# --- Ana Program ---
def main():
    print("=== ZED KAMERA KONTROL MERKEZİ (Nihai Sürüm) ===")
    zed_cam = ZEDCamera()
    try:
        if zed_cam.open_camera():
            zed_cam.yolo_volume_measurement()
    except Exception as e:
        print(f"\nBeklenmedik bir kritik hata oluştu: {e}")
    finally:
        zed_cam.close()
        print("Program sonlandırıldı.")

if __name__ == "__main__":
    main()