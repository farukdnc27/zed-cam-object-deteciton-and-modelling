import pyzed.sl as sl
import numpy as np
import cv2
import time
from ultralytics import YOLO
import math
import open3d as o3d

class ZEDVolumeEstimator:
    """
    ZED 2i kamera ve YOLOv8 kullanarak nesnelerin hacmini 3D olarak tahmin eden ve
    Open3D ile görselleştiren sınıf.
    Orijinal RealSense kodu, ZED SDK (pyzed) kullanacak şekilde tamamen yeniden yazılmıştır.
    """
    def __init__(self, width=1280, height=720, fps=30):
        print("ZED 2i Tabanlı Kapsamlı 3D Hacim Tahmini başlatılıyor...")

        self.model = YOLO('yolov8n.pt')
        print("YOLOv8 modeli yüklendi.")

        # --- ZED Kamera Yapıla ndırması ---
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720 # HD720 (1280x720) daha iyi sonuçlar verebilir
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Kalite odaklı mod
        self.init_params.coordinate_units = sl.UNIT.METER # Metre cinsinden çalış
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_DOWN # OpenCV/Görüntü koordinatlarına en yakın sistem

        # Kamerayı aç
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"ZED kamera açılamadı: {err}")
            exit(1)
        
        # Kamera iç parametrelerini al
        self.cam_info = self.zed.get_camera_information()
        self.intrinsics = self.cam_info.camera_configuration.calibration_parameters.left_cam
        print(f"ZED 2i kamera {width}x{height} @ {fps} FPS'de hazır.")

        # Tekrar kullanılacak ZED matrislerini oluştur
        self.color_mat = sl.Mat()
        self.point_cloud_mat = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()

        # --- Open3D Görselleştirici Yapılandırması ---
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("3D Sahne Gosterimi", width=800, height=600)
        self.pointcloud_o3d = o3d.geometry.PointCloud()
        
        self.is_first_frame = True # Kamerayı ilk karede ayarla
        print("Open3D görselleştirici hazır.")
        print("Çıkmak için 'ESC' tuşuna basın veya pencerelerden birini kapatın.")

    def _project_point_to_pixel(self, point_3d):
        """
        Verilen bir 3D noktayı (kamera koordinat sisteminde) 2D piksel koordinatlarına dönüştürür.
        ZED SDK'da doğrudan bir projeksiyon fonksiyonu olmadığından manuel olarak yapılır.
        """
        if point_3d[2] <= 0:  # Nokta kameranın arkasındaysa
            return None
        
        x_proj = (point_3d[0] * self.intrinsics.fx / point_3d[2]) + self.intrinsics.cx
        y_proj = (point_3d[1] * self.intrinsics.fy / point_3d[2]) + self.intrinsics.cy
        
        return (int(x_proj), int(y_proj))

    def stop(self):
        print("Uygulama kapatılıyor...")
        self.zed.close()
        self.vis.destroy_window()
        cv2.destroyAllWindows()

    def run(self):
        try:
            while True:
                # ZED kameradan yeni bir kare yakala
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    # Renkli görüntüyü ve nokta bulutunu al
                    self.zed.retrieve_image(self.color_mat, sl.VIEW.LEFT)
                    self.zed.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZRGBA)

                    # Verileri NumPy dizilerine dönüştür
                    color_image = self.color_mat.get_data()[:, :, :3] # Alfa kanalını at
                    point_cloud_data = self.point_cloud_mat.get_data()
                    
                    # --- Open3D Nokta Bulutu Güncellemesi ---
                    # ZED'den gelen XYZRGBA verisinden noktaları ve renkleri çıkar
                    points = point_cloud_data[:, :, :3].reshape(-1, 3)
                    
                    # Renkler float32 olarak paketlenmiştir, bunu 8-bit RGB'ye dönüştür
                    rgba_colors = point_cloud_data[:, :, 3].reshape(-1)
                    rgb_colors = np.frombuffer(rgba_colors.tobytes(), dtype=np.uint8).reshape(-1, 4)[:, :3] / 255.0

                    self.pointcloud_o3d.points = o3d.utility.Vector3dVector(points)
                    self.pointcloud_o3d.colors = o3d.utility.Vector3dVector(rgb_colors)

                    # YOLO ile nesne tespiti yap
                    results = self.model.predict(color_image, conf=0.5, verbose=False)
                    
                    geometries_to_add = [self.pointcloud_o3d]

                    for r in results:
                        for box in r.boxes:
                            try:
                                class_id = int(box.cls[0])
                                class_name = self.model.names[class_id]

                                # Sadece belirli sınıfları işle
                                if class_name not in ['bottle', 'cup', 'mouse', 'keyboard', 'laptop', 'cell phone', 'book']:
                                    continue

                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                center_x, center_y = int((x1+x2)/2), int((y1+y2)/2)

                                # --- ZED ile 3D Boyutlandırma ---
                                # ZED SDK, bir pikselin 3D koordinatını doğrudan verebilir. Bu, RealSense'in
                                # deprojection yönteminden daha basit ve genellikle daha doğrudur.
                                err_c, center_3d = self.point_cloud_mat.get_value(center_x, center_y)
                                err_l, p_left_3d = self.point_cloud_mat.get_value(x1, center_y)
                                err_r, p_right_3d = self.point_cloud_mat.get_value(x2, center_y)
                                err_t, p_top_3d = self.point_cloud_mat.get_value(center_x, y1)
                                err_b, p_bottom_3d = self.point_cloud_mat.get_value(center_x, y2)

                                # Geçerli 3D noktalar elde edilip edilmediğini kontrol et
                                if not all(math.isfinite(p[2]) for p in [center_3d, p_left_3d, p_right_3d, p_top_3d, p_bottom_3d]):
                                    continue

                                # 3D koordinatlardan genişlik ve yüksekliği hesapla
                                width_m = abs(p_right_3d[0] - p_left_3d[0])
                                height_m = abs(p_bottom_3d[1] - p_top_3d[1])

                                # Derinliği ve hacmi şekle göre tahmin et
                                if class_name in ['bottle', 'cup', 'vase', 'can']:
                                    # Silindirik nesne varsayımı
                                    depth_m = width_m 
                                    volume_m3 = (math.pi / 4) * width_m * depth_m * height_m
                                else:
                                    # Kutu şeklinde nesne varsayımı (derinlik = genişliğin yarısı gibi bir sezgisel)
                                    depth_m = width_m * 0.5 
                                    volume_m3 = width_m * height_m * depth_m
                                
                                # Open3D için yönlendirilmiş sınırlayıcı kutu oluştur
                                box_center_3d = np.array(center_3d[:3])
                                box_center_3d[2] -= depth_m / 2.0 # Kutunun merkezini nesnenin ortasına taşı

                                oriented_box = o3d.geometry.OrientedBoundingBox(
                                    center=box_center_3d,
                                    R=np.identity(3), # Eksen hizalı kutu
                                    extent=[width_m, height_m, depth_m]
                                )
                                oriented_box.color = (1, 0, 0) # Kırmızı renk
                                geometries_to_add.append(oriented_box)

                                # --- AR Kutusunu Çizmek İçin 3D Kutuyu 2D'ye Yansıt ---
                                box_corners_3d = np.asarray(oriented_box.get_box_points())
                                points_2d = [self._project_point_to_pixel(p) for p in box_corners_3d]
                                
                                # Sadece geçerli (ekran içinde) noktaları al
                                points_2d_valid = [p for p in points_2d if p is not None]
                                if len(points_2d_valid) == 8:
                                    # AR Kutusunu Çiz
                                    cv2.line(color_image, points_2d[0], points_2d[1], (0, 255, 0), 2); cv2.line(color_image, points_2d[1], points_2d[3], (0, 255, 0), 2); cv2.line(color_image, points_2d[3], points_2d[2], (0, 255, 0), 2); cv2.line(color_image, points_2d[2], points_2d[0], (0, 255, 0), 2)
                                    cv2.line(color_image, points_2d[4], points_2d[5], (0, 0, 255), 2); cv2.line(color_image, points_2d[5], points_2d[7], (0, 0, 255), 2); cv2.line(color_image, points_2d[7], points_2d[6], (0, 0, 255), 2); cv2.line(color_image, points_2d[6], points_2d[4], (0, 0, 255), 2)
                                    cv2.line(color_image, points_2d[0], points_2d[4], (255, 0, 0), 2); cv2.line(color_image, points_2d[1], points_2d[5], (255, 0, 0), 2); cv2.line(color_image, points_2d[2], points_2d[6], (255, 0, 0), 2); cv2.line(color_image, points_2d[3], points_2d[7], (255, 0, 0), 2)

                                # Bilgi metnini yazdır
                                volume_text = f"{volume_m3 * 1e6 / 1000:.2f} L" if volume_m3 * 1e6 >= 1000 else f"{volume_m3 * 1e6:.0f} ml"
                                info_text = f"{class_name}: {volume_text} (W:{width_m*100:.1f}cm)"
                                cv2.putText(color_image, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                                cv2.putText(color_image, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                            
                            except Exception as e:
                                # print(f"Hata: {e}") # Hata ayıklama için
                                continue
                
                # --- Görselleştirme ---
                self.vis.clear_geometries()
                for geom in geometries_to_add:
                    self.vis.add_geometry(geom, reset_bounding_box=self.is_first_frame)
                
                self.is_first_frame = False # İlk kareden sonra görünümü sabitle

                keep_running = self.vis.poll_events()
                self.vis.update_renderer()
                
                cv2.imshow("ZED - 2D AR Gorunumu", color_image)
                key = cv2.waitKey(1)
                
                if key == 27 or not keep_running:
                    break
        finally:
            self.stop()

if __name__ == "__main__":
    estimator = ZEDVolumeEstimator()
    estimator.run()