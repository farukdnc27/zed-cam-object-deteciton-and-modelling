import pyzed.sl as sl
import cv2
import numpy as np
import math
from typing import List, Optional, Tuple
import random

class EdgeDetector:
    def __init__(self):
        # ZED kamera nesneleri
        self.zed = sl.Camera()
        self.image_sl = sl.Mat()
        self.depth_map = sl.Mat()
        self.xyz_map = sl.Mat()

        # Edge detection parametreleri
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.min_contour_area = 100
        self.approx_poly_epsilon = 0.02
        
        # Görüntü nesneleri
        self.contour_image = None
        self.edges_image = None
        self.debug_image = None
        self.watershed_image = None
        
        # ROI (Region of Interest) seçimi
        self.roi = None
        self.selecting_roi = False

    def initialize_camera(self) -> bool:
        """ZED kamerasını başlat"""
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30 
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"ZED kamera başlatılamadı: {err}")
            return False

        print("ZED kamera başarıyla başlatıldı")
        print("Kullanım:")
        print("- 'q' veya ESC: Çıkış")
        print("- 's': ROI seçme modu")
        print("- 'r': ROI'yi sıfırla")
        print("- 'w': Watershed görünümünü aç/kapat")
        return True

    def classify_shape(self, approx: np.ndarray, area: float) -> str:
        """Konturun şeklini sınıflandır"""
        vertices = len(approx)
        
        if vertices == 3:
            return "Üçgen"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            return "Kare" if 0.9 <= aspect_ratio <= 1.1 else "Dikdörtgen"
        elif vertices == 5:
            return "Beşgen"
        elif vertices == 6:
            return "Altıgen"
        else:
            perimeter = cv2.arcLength(approx, True)
            if perimeter == 0:
                return "Bilinmeyen"
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            return "Daire" if circularity > 0.8 else f"{vertices}-gen"

    def get_3d_coordinates(self, x: int, y: int) -> Optional[List[float]]:
        """Belirtilen pikselin 3D koordinatlarını döndürür"""
        try:
            x, y = int(x), int(y)
            if not (0 <= x < self.xyz_map.get_width() and 0 <= y < self.xyz_map.get_height()):
                return None
            
            status, point3d = self.xyz_map.get_value(x, y)
            
            if status != sl.ERROR_CODE.SUCCESS or math.isnan(point3d[0]):
                return None
            
            return [point3d[0], point3d[1], point3d[2]]
            
        except Exception as e:
            print(f"3D koordinat alma hatası: {e}")
            return None

    def calculate_3d_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Optional[float]:
        """İki piksel arasındaki 3D mesafeyi hesapla"""
        try:
            pt1 = self.get_3d_coordinates(p1[0], p1[1])
            pt2 = self.get_3d_coordinates(p2[0], p2[1])
            
            if pt1 is None or pt2 is None:
                return None
                
            distance = math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 + (pt2[2]-pt1[2])**2)
            return distance
            
        except Exception as e:
            print(f"3D mesafe hesaplama hatası: {e}")
            return None

    def draw_measurements(self, approx: np.ndarray, image: np.ndarray, color: tuple):
        """Kenar uzunluklarını ve şekil bilgilerini çiz"""
        try:
            for i in range(len(approx)):
                p1 = (int(approx[i][0][0]), int(approx[i][0][1]))
                p2 = (int(approx[(i + 1) % len(approx)][0][0]), int(approx[(i + 1) % len(approx)][0][1]))
                
                distance_m = self.calculate_3d_distance(p1, p2)
                if distance_m is None:
                    continue
                    
                length_cm = distance_m * 100
                mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                
                cv2.line(image, p1, p2, color, 2)
                cv2.putText(image, f"{length_cm:.1f}cm", mid_point,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(image, p1, 5, color, -1)
                
                pt3d = self.get_3d_coordinates(p1[0], p1[1])
                if pt3d:
                    info_text = f"({pt3d[0]:.2f}, {pt3d[1]:.2f}, {pt3d[2]:.2f})m"
                    cv2.putText(image, info_text, (p1[0], p1[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
        except Exception as e:
            print(f"Ölçüm çizme hatası: {e}")

    def separate_overlapping_objects(self, image: np.ndarray) -> np.ndarray:
        """Watershed algoritmasıyla çakışan nesneleri ayır"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]
        
        return markers

    def process_contours(self, contours: List[np.ndarray], image: np.ndarray):
        """Konturları işle ve şekilleri çiz"""
        self.contour_image = image.copy()
        self.debug_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.debug_image = cv2.cvtColor(self.debug_image, cv2.COLOR_GRAY2BGR)
        
        # Konturları alana göre sırala
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for i, contour in enumerate(contours):
            try:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue
                    
                epsilon = self.approx_poly_epsilon * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                shape_type = self.classify_shape(approx, area)
                
                # Her nesne için rastgele renk
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                # Kontur ve bounding box çiz
                cv2.drawContours(self.contour_image, [contour], -1, color, 2)
                cv2.drawContours(self.debug_image, [contour], -1, color, 2)
                
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(self.debug_image, (x, y), (x+w, y+h), color, 2)
                
                # Merkeze ID ve şekil bilgisi yaz
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(self.contour_image, f"{i}: {shape_type}", (cx, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(self.debug_image, f"ID:{i} Area:{area:.0f}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                # Ölçümleri çiz
                self.draw_measurements(approx, self.contour_image, color)
                self.draw_measurements(approx, self.debug_image, color)
                
            except Exception as e:
                print(f"Kontur {i} işleme hatası: {e}")
                continue

    def process_frame(self):
        """Her kareyi işle"""
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_sl, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            self.zed.retrieve_measure(self.xyz_map, sl.MEASURE.XYZ)
            
            image_left = self.image_sl.get_data()
            gray = cv2.cvtColor(image_left, cv2.COLOR_BGRA2GRAY)
            
            if self.roi is not None:
                x, y, w, h = self.roi
                gray = gray[y:y+h, x:x+w]
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
            self.edges_image = cv2.Canny(blurred, 
                                      self.canny_low_threshold, 
                                      self.canny_high_threshold)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.edges_image = cv2.morphologyEx(self.edges_image, 
                                             cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(self.edges_image, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
            
            if self.roi is not None:
                x_offset, y_offset = self.roi[0], self.roi[1]
                for contour in contours:
                    contour += np.array([[x_offset, y_offset]])
            
            bgr = image_left[:, :, :3]
            self.process_contours(contours, bgr)
            self.watershed_image = self.separate_overlapping_objects(bgr.copy())

    def setup_ui(self):
        """Kullanıcı arayüzünü kur"""
        self.window_name = "ZED 3D Ölçüm"
        cv2.namedWindow(self.window_name)
        
        def nothing(x): pass
        
        cv2.createTrackbar("Canny Low", self.window_name, 
                         self.canny_low_threshold, 200, nothing)
        cv2.createTrackbar("Canny High", self.window_name, 
                         self.canny_high_threshold, 300, nothing)
        cv2.createTrackbar("Min Alan", self.window_name, 
                         self.min_contour_area, 1000, nothing)
        cv2.createTrackbar("Epsilon", self.window_name, 
                         int(self.approx_poly_epsilon*100), 100, nothing)

    def update_parameters(self):
        """Parametreleri güncelle"""
        self.canny_low_threshold = cv2.getTrackbarPos("Canny Low", self.window_name)
        self.canny_high_threshold = cv2.getTrackbarPos("Canny High", self.window_name)
        self.min_contour_area = cv2.getTrackbarPos("Min Alan", self.window_name)
        self.approx_poly_epsilon = cv2.getTrackbarPos("Epsilon", self.window_name)/100.0

    def run(self):
        """Ana çalışma döngüsü"""
        if not self.initialize_camera():
            return
            
        self.setup_ui()
        show_watershed = False
        
        while True:
            self.update_parameters()
            self.process_frame()
            
            if self.contour_image is not None:
                cv2.imshow(self.window_name, self.contour_image)
            if self.debug_image is not None:
                cv2.imshow("Debug Görünümü", self.debug_image)
            if show_watershed and self.watershed_image is not None:
                cv2.imshow("Watershed", self.watershed_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                print("ROI seçme modu - lütfen bir bölge seçin")
                self.roi = cv2.selectROI(self.window_name, self.contour_image)
                print(f"Seçilen ROI: {self.roi}")
            elif key == ord('r'):
                self.roi = None
                print("ROI sıfırlandı")
            elif key == ord('w'):
                show_watershed = not show_watershed
                if not show_watershed:
                    cv2.destroyWindow("Watershed")
            
        self.zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EdgeDetector()
    detector.run()