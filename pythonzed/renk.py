import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

class MultiCameraObjectAnalyzer:
    def __init__(self, camera_type='realsense'):
        self.camera_type = camera_type
        self.setup_camera()
        
        # Ortak ayarlar
        self.color_ranges = {
            "kırmızı": ([0, 100, 100], [10, 255, 255]),
            "mavi": ([100, 100, 100], [130, 255, 255]),
            "yeşil": ([40, 40, 40], [80, 255, 255])
        }
        self.min_object_size = 1000
        self.mm_per_pixel = None
        
    def setup_camera(self):
        if self.camera_type == 'realsense':
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
        else:  # Webcam/IP Camera
            self.cap = cv2.VideoCapture(0 if self.camera_type == 'webcam' else self.camera_type)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def get_frames(self):
        if self.camera_type == 'realsense':
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            return np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data())
        else:
            ret, frame = self.cap.read()
            return frame if ret else None, None
    
    def calibrate(self, color_image, depth_image=None):
        print("Kalibrasyon için referans nesnenin İKİ UCUNU tıklayın (ESC ile kaydeder)")
        points = []
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                cv2.circle(color_image, (x,y), 5, (0,0,255), -1)
                points.append((x,y))
                if len(points) == 2:
                    if self.camera_type == 'realsense' and depth_image is not None:
                        depth1 = depth_image[points[0][1], points[0][0]] / 1000
                        depth2 = depth_image[points[1][1], points[1][0]] / 1000
                        pixel_dist = np.sqrt((points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2)
                        self.mm_per_pixel = 100 / (pixel_dist * (depth1+depth2)/2)  # 100mm referans
                    else:
                        pixel_dist = np.sqrt((points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2)
                        self.mm_per_pixel = 100 / pixel_dist  # Webcam için sabit ölçek
                    print(f"Kalibrasyon tamam: 1 piksel = {self.mm_per_pixel:.4f} mm")
        
        cv2.namedWindow("Kalibrasyon")
        cv2.setMouseCallback("Kalibrasyon", click_event)
        
        while True:
            cv2.imshow("Kalibrasyon", color_image)
            key = cv2.waitKey(1)
            if key == 27 or len(points) >= 2:  # ESC
                break
        cv2.destroyAllWindows()
        return len(points) == 2
    
    def detect_objects(self, color_image, depth_image=None):
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        detected_objects = []
        
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < self.min_object_size:
                    continue
                
                x,y,w,h = cv2.boundingRect(cnt)
                cx, cy = x+w//2, y+h//2
                
                obj_data = {
                    "color": color_name,
                    "bbox": (x,y,w,h),
                    "center": (cx, cy),
                    "contour": cnt,
                    "depth": -1,
                    "dimensions": None
                }
                
                if depth_image is not None:
                    obj_data["depth"] = depth_image[cy, cx] / 1000
                    if self.mm_per_pixel:
                        obj_data["dimensions"] = self.calculate_dimensions(cnt, depth_image)
                
                detected_objects.append(obj_data)
        
        return detected_objects
    
    def calculate_dimensions(self, contour, depth_image):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        dimensions = []
        for i in range(4):
            x,y = box[i]
            nx,ny = box[(i+1)%4]
            depth = depth_image[y,x] / 1000
            pixel_length = np.sqrt((nx-x)**2 + (ny-y)**2)
            real_length = pixel_length * depth * self.mm_per_pixel
            dimensions.append(real_length)
        
        return {
            "width": (dimensions[0] + dimensions[2]) / 2,
            "height": (dimensions[1] + dimensions[3]) / 2,
            "depth": np.mean([depth_image[p[1],p[0]] for p in box]) / 1000
        }
    
    def visualize(self, color_image, objects):
        output = color_image.copy()
        for obj in objects:
            x,y,w,h = obj["bbox"]
            cv2.rectangle(output, (x,y), (x+w,y+h), (0,255,0), 2)
            
            info = f"{obj['color']}"
            if obj['depth'] > 0:
                info += f" {obj['depth']:.2f}m"
            if obj['dimensions']:
                dim = obj['dimensions']
                info += f" {dim['width']:.1f}x{dim['height']:.1f}mm"
            
            cv2.putText(output, info, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            
            cv2.drawContours(output, [obj["contour"]], -1, (0,0,255), 1)
            cv2.circle(output, obj["center"], 3, (255,0,0), -1)
        
        return output
    
    def run(self):
        try:
            # İlk kalibrasyon
            color, depth = self.get_frames()
            if color is None:
                print("Kamera bağlantı hatası!")
                return
            
            if not self.calibrate(color.copy(), depth):
                print("Kalibrasyon başarısız!")
                return
            
            # Ana döngü
            while True:
                color, depth = self.get_frames()
                if color is None:
                    continue
                
                objects = self.detect_objects(color, depth)
                output = self.visualize(color, objects)
                
                cv2.imshow("Nesne Analiz Sistemi", output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            if self.camera_type == 'realsense':
                self.pipeline.stop()
            else:
                self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Kamera Seçin:")
    print("1 - Intel RealSense (Derinlik + Renk)")
    print("2 - Standart Webcam (Sadece Renk)")
    
    choice = input("Seçiminiz (1/2): ")
    
    if choice == "1":
        analyzer = MultiCameraObjectAnalyzer('realsense')
    else:
        analyzer = MultiCameraObjectAnalyzer('webcam')
    
    analyzer.run()