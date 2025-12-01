import pyzed.sl as sl
import cv2
import numpy as np
import math
from typing import List, Tuple

class Auto3DMeasurer:
    def __init__(self):
        # Kamera ayarları
        self.zed = sl.Camera()
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.xyz = sl.Mat()
        
        # Ölçüm parametreleri
        self.min_object_size = 0.1  # metre (10cm'den küçük nesneleri görmezden gel)
        self.max_distance = 3.0     # metre (3m'den uzak nesneleri ölçme)
        self.detection_threshold = 0.5  # Derinlik süreksizliği eşiği (metre)

    def initialize_camera(self) -> bool:
        """ZED kamerasını başlat"""
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Kamera başlatılamadı: {err}")
            return False
        return True

    def get_image_data(self) -> np.ndarray:
        """ZED görüntüsünü OpenCV formatına dönüştür"""
        image_data = self.image.get_data()
        return image_data[:, :, :3].copy()  # BGR formatında kopya döndür

    def detect_edges_3d(self) -> List[Tuple[float, Tuple[float, float, float]]]:
        """Otomatik kenar tespiti ve 3D koordinat dönüşü"""
        edges_3d = []
        
        # Derinlik haritasındaki ani değişimleri bul
        depth_map = self.depth.get_data()
        if depth_map is None:
            return edges_3d
            
        grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        edge_mask = (np.sqrt(grad_x**2 + grad_y**2) > self.detection_threshold)
        
        # Kenar piksellerinin 3D koordinatlarını al
        for y in range(0, edge_mask.shape[0], 5):  # 5 piksel atlayarak optimize
            for x in range(0, edge_mask.shape[1], 5):
                if edge_mask[y,x]:
                    status, point = self.xyz.get_value(x, y)
                    if status == sl.ERROR_CODE.SUCCESS and not math.isnan(point[0]):
                        if 0 < point[2] < self.max_distance:  # Z ekseni kontrolü
                            edges_3d.append((point[2], (point[0], point[1], point[2])))
        
        # Derinliğe göre sırala (en yakın nesneler önce)
        edges_3d.sort()
        return [coord for _, coord in edges_3d]

    def group_points_to_objects(self, points: List[Tuple[float, float, float]]) -> List[List[Tuple[float, float, float]]]:
        """3D noktaları nesnelere grupla"""
        objects = []
        while points:
            # Bir başlangıç noktası seç
            current_obj = [points.pop(0)]
            
            # Yakın noktaları bul ve ekle
            i = 0
            while i < len(points):
                if self._distance_3d(current_obj[-1], points[i]) < self.min_object_size:
                    current_obj.append(points.pop(i))
                else:
                    i += 1
            
            if len(current_obj) > 3:  # En az 4 noktalı grupları nesne olarak kabul et
                objects.append(current_obj)
        return objects

    def _distance_3d(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """İki 3D nokta arası mesafe"""
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)

    def calculate_object_dimensions(self, object_points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Nesnenin boyutlarını hesapla (en, boy, yükseklik)"""
        points = np.array(object_points)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        return tuple(max_coords - min_coords)

    def run_auto_measurements(self):
        """Tam otomatik ölçüm döngüsü"""
        if not self.initialize_camera():
            return

        cv2.namedWindow("Otomatik 3D Ölçüm")
        print("Otomatik ölçüm başladı. ESC ile çıkış yapabilirsiniz.")

        while True:
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Verileri al
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
                self.zed.retrieve_measure(self.xyz, sl.MEASURE.XYZ)
                
                # Görüntüyü doğru formatta al
                frame = self.get_image_data()
                if frame is None:
                    continue
                
                # 1. Kenarları tespit et
                edge_points = self.detect_edges_3d()
                
                # 2. Noktaları nesnelere grupla
                objects = self.group_points_to_objects(edge_points)
                
                # 3. Her nesne için ölçüm yap
                for i, obj in enumerate(objects[:3]):  # En fazla 3 nesne göster
                    try:
                        # Boyutları hesapla
                        width, height, depth = self.calculate_object_dimensions(obj)
                        
                        # Merkez noktasını bul
                        center = np.mean(obj, axis=0)
                        
                        # Görüntüye yazdır
                        text = f"Obj {i+1}: {width*100:.1f}x{height*100:.1f}x{depth*100:.1f} cm"
                        cv2.putText(frame, text, (20, 30*(i+1)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                        
                        # Konsola yazdır
                        print(f"{text} | Position: X={center[0]:.2f}m Y={center[1]:.2f}m Z={center[2]:.2f}m")
                        
                    except Exception as e:
                        print(f"Measurement error: {e}")
                        continue
                
                # Görüntüyü göster
                cv2.imshow("Otomatik 3D Ölçüm", frame)
            
            if cv2.waitKey(10) == 27:  # ESC
                break

        self.zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    measurer = Auto3DMeasurer()
    measurer.run_auto_measurements()