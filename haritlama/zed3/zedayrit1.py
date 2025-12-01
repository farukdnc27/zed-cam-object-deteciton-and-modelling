import pyzed.sl as sl
import cv2
import numpy as np
import math

class EdgeDetector:
    def __init__(self):
        # ZED kamera ve görüntü nesnelerini başlat
        self.zed = sl.Camera()
        self.image_sl = sl.Mat()

        # Kenar tespiti parametreleri
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.min_contour_area = 100

        # Sonuç görüntüleri için değişkenler
        self.contour_image = None
        self.edges_image = None

    def initialize(self):
        """ZED kamerasını başlatır ve ayarlarını yapar."""
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"ZED açılamadı: {err}")
            return False
            
        print("ZED Edge Detection başlatıldı...")
        print("Çıkmak için 'q' tuşuna basın")
        return True

    def classify_shape(self, approx, area):
        """Kontur yaklaşımına göre şekli sınıflandırır."""
        vertices = len(approx)
        
        if vertices == 3:
            return "Ucgen"
        elif vertices == 4:
            # Dikdörtgen/kare kontrolü için sınırlayıcı kutu kullan
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "Kare"
            else:
                return "Dikdortgen"
        elif vertices == 5:
            return "Besgen"
        elif vertices == 6:
            return "Altigen"
        else:
            # Daire kontrolü için dairesellik (circularity) hesapla
            perimeter = cv2.arcLength(approx, True)
            if perimeter == 0:
                return "Bilinmeyen"
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity > 0.75:
                return "Daire"
            else:
                return "Cokgen"

        return "Bilinmeyen"

    def show_edge_lengths(self, approx, image):
        """Tespit edilen şeklin kenar uzunluklarını görüntü üzerinde gösterir."""
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            
            # Kenar uzunluğunu piksel cinsinden hesapla
            length = cv2.norm(p1, p2)
            
            # Kenarın orta noktası
            mid_point = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
            
            # Uzunluğu yazdır
            length_text = f"{int(length)}px"
            cv2.putText(image, length_text, mid_point, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Kenarı farklı bir renkle vurgula
            cv2.line(image, tuple(p1), tuple(p2), (255, 0, 255), 2)


    def detect_and_draw_shapes(self, contours, source_image):
        """Konturları analiz eder, şekilleri tespit eder ve çizer."""
        self.contour_image = source_image.copy() # Orijinal görüntüyü kopyala
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Küçük alanlı konturları filtrele
            if area < self.min_contour_area:
                continue
                
            # Konturu daha basit bir çokgene yaklaştır
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Şekil tipini belirle
            shape_type = self.classify_shape(approx, area)
            
            # Konturu çiz
            cv2.drawContours(self.contour_image, [contour], -1, (0, 255, 0), 2)
            
            # Köşe noktalarını daire ile işaretle
            for point in approx:
                cv2.circle(self.contour_image, tuple(point[0]), 5, (0, 0, 255), -1)
                
            # Şekil ismini merkeze yaz
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(self.contour_image, shape_type, (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            
            # Kenar uzunluklarını göster
            self.show_edge_lengths(approx, self.contour_image)

    def process_frame(self):
        """Kameradan tek bir kare alır ve işler."""
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Sol görüntüyü al
            self.zed.retrieve_image(self.image_sl, sl.VIEW.LEFT)
            # OpenCV (NumPy) formatına dönüştür
            image_left_ocv = self.image_sl.get_data()
            
            # 4 kanallı BGRA'dan gri tonlamalıya çevir
            gray_image = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGRA2GRAY)
            
            # Gürültüyü azaltmak için Gauss filtresi uygula
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 1.5)
            
            # Canny kenar tespiti
            self.edges_image = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
            
            # Morfolojik operasyonlar ile kenarları birleştir/iyileştir
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.edges_image = cv2.morphologyEx(self.edges_image, cv2.MORPH_CLOSE, kernel)
            
            # Konturları bul
            contours, _ = cv2.findContours(self.edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Geometrik şekilleri tespit et ve çiz (orijinal renkli görüntü üzerine)
            # 4 kanallı görüntüden son 3 kanalı (BGR) alarak çizim yapabiliriz
            # ya da 4 kanallı görüntü üzerinde çalışmaya devam edebiliriz.
            bgr_image = image_left_ocv[:, :, :3]
            self.detect_and_draw_shapes(contours, bgr_image)
            
    def setup_ui(self):
        """OpenCV pencerelerini ve trackbar'ları oluşturur."""
        self.window_name = "ZED - Geometrik Sekil Tespiti"
        cv2.namedWindow(self.window_name)
        
        # C++'daki gibi &değişken yerine callback kullanılır.
        # Değerler döngü içinde getTrackbarPos ile okunacağı için boş lambda yeterlidir.
        def nothing(x):
            pass

        cv2.createTrackbar("Canny Low", self.window_name, self.canny_low_threshold, 200, nothing)
        cv2.createTrackbar("Canny High", self.window_name, self.canny_high_threshold, 300, nothing)
        cv2.createTrackbar("Min Alan", self.window_name, self.min_contour_area, 1000, nothing)

    def update_params_from_trackbar(self):
        """Trackbar'lardan güncel parametre değerlerini okur."""
        self.canny_low_threshold = cv2.getTrackbarPos("Canny Low", self.window_name)
        self.canny_high_threshold = cv2.getTrackbarPos("Canny High", self.window_name)
        self.min_contour_area = cv2.getTrackbarPos("Min Alan", self.window_name)

    def run(self):
        """Ana uygulama döngüsü."""
        if not self.initialize():
            return
            
        self.setup_ui()

        while True:
            self.update_params_from_trackbar()
            self.process_frame()
            
            # Sonuçları göster
            if self.contour_image is not None:
                cv2.imshow(self.window_name, self.contour_image)
            if self.edges_image is not None:
                cv2.imshow("Kenarlar", self.edges_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' veya ESC tuşu
                break
        
        self.zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = EdgeDetector()
    detector.run()