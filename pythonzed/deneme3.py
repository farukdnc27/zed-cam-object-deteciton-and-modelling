import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VolumeCalculator:
    def __init__(self):
        # RealSense pipeline'ını başlat
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Depth ve color stream'lerini etkinleştir
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Align objesi - color ve depth görüntülerini hizalamak için
        self.align = rs.align(rs.stream.color)
        
        # Filtreleme için
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        
        self.intrinsics = None
        
    def start_camera(self):
        """Kamerayı başlat ve intrinsic parametreleri al"""
        try:
            profile = self.pipeline.start(self.config)
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            self.intrinsics = depth_profile.get_intrinsics()
            print(f"Kamera başlatıldı. Çözünürlük: {self.intrinsics.width}x{self.intrinsics.height}")
            return True
        except Exception as e:
            print(f"Kamera başlatılamadı: {e}")
            return False
    
    def stop_camera(self):
        """Kamerayı durdur"""
        self.pipeline.stop()
    
    def capture_frame(self):
        """Bir frame yakala ve işle"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        # Depth filtreleme
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        
        # NumPy array'e çevir
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return depth_image, color_image, depth_frame
    
    def depth_to_point_cloud(self, depth_frame, roi=None):
        """Depth frame'i 3D nokta bulutuna çevir"""
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        
        # Nokta koordinatlarını al
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # ROI (Region of Interest) uygulanırsa
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            # Pixel koordinatlarını kullanarak filtreleme
            mask = np.zeros(len(vertices), dtype=bool)
            for i, vertex in enumerate(vertices):
                # 3D noktayı pixel koordinatına çevir
                pixel = rs.rs2_project_point_to_pixel(self.intrinsics, vertex)
                if x_min <= pixel[0] <= x_max and y_min <= pixel[1] <= y_max:
                    mask[i] = True
            vertices = vertices[mask]
        
        # Geçersiz noktaları temizle
        valid_mask = ~np.isnan(vertices).any(axis=1) & ~np.isinf(vertices).any(axis=1)
        vertices = vertices[valid_mask]
        
        # Z değeri 0 olan noktaları kaldır
        vertices = vertices[vertices[:, 2] > 0]
        
        return vertices
    
    def calculate_volume_convex_hull(self, points):
        """Convex Hull kullanarak hacim hesapla"""
        if len(points) < 4:
            return 0
        
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception as e:
            print(f"Convex Hull hesaplanamadı: {e}")
            return 0
    
    def calculate_volume_voxel(self, points, voxel_size=0.001):
        """Voxel tabanlı hacim hesaplama"""
        if len(points) == 0:
            return 0
        
        # Open3D nokta bulutu oluştur
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Voxel grid oluştur
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
        
        # Hacim = voxel sayısı * voxel hacmi
        volume = len(voxel_grid.get_voxels()) * (voxel_size ** 3)
        return volume
    
    def filter_outliers(self, points, nb_neighbors=20, std_ratio=2.0):
        """Outlier noktaları filtrele"""
        if len(points) == 0:
            return points
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        return np.asarray(pcd.points)
    
    def visualize_point_cloud(self, points, title="Nokta Bulutu"):
        """3D nokta bulutunu görselleştir"""
        if len(points) == 0:
            print("Görselleştirilecek nokta yok")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        plt.show()
    
    def interactive_roi_selection(self, color_image):
        """Kullanıcıdan ROI seçmesini iste"""
        print("Hacim hesaplanacak bölgeyi seçin (sol tık + sürükle)")
        roi = cv2.selectROI("ROI Seçimi", color_image, False)
        cv2.destroyWindow("ROI Seçimi")
        
        if roi[2] > 0 and roi[3] > 0:  # Geçerli ROI seçildi
            return (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
        return None

def main():
    calculator = VolumeCalculator()
    
    if not calculator.start_camera():
        return
    
    print("\nKomutlar:")
    print("'c' - Hacim hesapla")
    print("'r' - ROI ile hacim hesapla")
    print("'v' - Nokta bulutunu görselleştir")
    print("'s' - Anlık görüntü kaydet")
    print("'q' - Çıkış")
    
    try:
        current_points = None
        
        while True:
            # Frame yakala
            depth_image, color_image, depth_frame = calculator.capture_frame()
            
            if depth_image is None:
                continue
            
            # Görüntüleri göster
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            
            cv2.imshow('RealSense - Color | Depth', images)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Tüm görüş alanı için hacim hesapla
                print("Nokta bulutu oluşturuluyor...")
                points = calculator.depth_to_point_cloud(depth_frame)
                
                if len(points) > 0:
                    print(f"Toplam nokta sayısı: {len(points)}")
                    
                    # Outlier filtreleme
                    points = calculator.filter_outliers(points)
                    print(f"Filtrelemeden sonra nokta sayısı: {len(points)}")
                    
                    # Hacim hesapla
                    volume_hull = calculator.calculate_volume_convex_hull(points)
                    volume_voxel = calculator.calculate_volume_voxel(points)
                    
                    print(f"\n--- Hacim Sonuçları ---")
                    print(f"Convex Hull Hacmi: {volume_hull:.6f} m³ ({volume_hull*1000000:.2f} cm³)")
                    print(f"Voxel Hacmi: {volume_voxel:.6f} m³ ({volume_voxel*1000000:.2f} cm³)")
                    
                    current_points = points
                else:
                    print("Geçerli nokta bulunamadı!")
            
            elif key == ord('r'):
                # ROI seçerek hacim hesapla
                roi = calculator.interactive_roi_selection(color_image.copy())
                
                if roi:
                    print(f"Seçilen ROI: {roi}")
                    print("ROI için nokta bulutu oluşturuluyor...")
                    
                    points = calculator.depth_to_point_cloud(depth_frame, roi)
                    
                    if len(points) > 0:
                        print(f"ROI içindeki nokta sayısı: {len(points)}")
                        
                        # Outlier filtreleme
                        points = calculator.filter_outliers(points)
                        print(f"Filtrelemeden sonra nokta sayısı: {len(points)}")
                        
                        # Hacim hesapla
                        volume_hull = calculator.calculate_volume_convex_hull(points)
                        volume_voxel = calculator.calculate_volume_voxel(points)
                        
                        print(f"\n--- ROI Hacim Sonuçları ---")
                        print(f"Convex Hull Hacmi: {volume_hull:.6f} m³ ({volume_hull*1000000:.2f} cm³)")
                        print(f"Voxel Hacmi: {volume_voxel:.6f} m³ ({volume_voxel*1000000:.2f} cm³)")
                        
                        current_points = points
                    else:
                        print("ROI içinde geçerli nokta bulunamadı!")
            
            elif key == ord('v'):
                # Nokta bulutunu görselleştir
                if current_points is not None:
                    calculator.visualize_point_cloud(current_points, "3D Nokta Bulutu")
                else:
                    print("Önce hacim hesaplama yapın!")
            
            elif key == ord('s'):
                # Anlık görüntü kaydet
                timestamp = cv2.getTickCount()
                cv2.imwrite(f'color_{timestamp}.png', color_image)
                cv2.imwrite(f'depth_{timestamp}.png', depth_colormap)
                print(f"Görüntüler kaydedildi: color_{timestamp}.png, depth_{timestamp}.png")
    
    except KeyboardInterrupt:
        print("\nProgram durduruluyor...")
    
    finally:
        calculator.stop_camera()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()