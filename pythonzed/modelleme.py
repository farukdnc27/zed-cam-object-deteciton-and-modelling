import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt

class DepthCamera:
    """
    Intel RealSense Derinlik Kamerasını yönetmek için bir sınıf.
    Kamerayı yapılandırır, veri akışını başlatır ve kareleri alır.
    """
    def __init__(self, resolution_width, resolution_height):
        # Derinlik ve renk akışlarını yapılandır
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Desteklenen bir çözünürlük ayarlamak için cihaz ürün hattını al
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        
        # Cihazın derinlik ölçeğini al
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Hizalama nesnesi oluştur (derinlik karesini renk karesiyle hizalamak için)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # Akışları etkinleştir
        config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, 30)
        
        # Veri akışını başlat
        self.profile = self.pipeline.start(config)

    def get_raw_frame(self):
        """
        Kameradan hizalanmış derinlik ve renk karelerini alır.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return False, None, None, None
            
        return True, depth_frame, color_frame

    def get_depth_scale(self):
        """
        Derinlik haritası birimleri ile metre arasındaki ilişkiyi döndürür.
        """
        return self.depth_scale

    def release(self):
        """
        Kamera kaynaklarını serbest bırakır.
        """
        self.pipeline.stop()

def depth2PointCloud(depth, rgb, intrinsics, depth_scale, clip_distance_max):
    """
    Derinlik ve renkli görüntü verilerini 3D nokta bulutuna dönüştürür.
    
    :param depth: Derinlik karesi (pyrealsense2.depth_frame)
    :param rgb: Renk karesi (pyrealsense2.video_frame)
    :param intrinsics: Kamera içsel parametreleri
    :param depth_scale: Derinlik ölçeği (genellikle 0.001)
    :param clip_distance_max: Belirtilen metrenin üzerindeki değerleri kırp
    :return: XYZ ve RGB verilerini içeren numpy dizisi
    """
    # Görüntü verilerini numpy dizilerine dönüştür
    depth_image = np.asanyarray(depth.get_data())
    rgb_image = np.asanyarray(rgb.get_data())
    
    rows, cols = depth_image.shape
    
    # Derinlik değerlerini metreye dönüştür
    depth_image = depth_image.astype(float) * depth_scale

    # Piksel koordinatlarını oluştur
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

    # Geçerli pikselleri belirle (0'dan büyük ve kırpma mesafesinden küçük olanlar)
    valid = (depth_image > 0) & (depth_image < clip_distance_max)
    
    # 3D noktaları hesapla
    x = np.where(valid, (c - intrinsics.ppx) / intrinsics.fx * depth_image, 0)
    y = np.where(valid, (r - intrinsics.ppy) / intrinsics.fy * depth_image, 0)
    z = np.where(valid, depth_image, 0)
    
    # Sadece geçerli piksellerin XYZ ve RGB değerlerini al
    points_xyz = np.dstack((x, y, z)).reshape(-1, 3)
    colors_rgb = rgb_image.reshape(-1, 3)
    
    points_xyzrgb = np.hstack((points_xyz[valid.flatten()], colors_rgb[valid.flatten()]))
    
    return points_xyzrgb

def create_point_cloud_file(vertices, filename):
    """
    Nokta bulutu verilerini bir PLY dosyasına yazar.
    
    :param vertices: Nokta bulutu verileri (x, y, z, r, g, b)
    :param filename: Kaydedilecek dosyanın adı (örn. "output.ply")
    """
    ply_header = f'''ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename, 'w') as f:
        f.write(ply_header)
        # Verileri tamsayı renk değerleriyle doğru formatta yaz
        np.savetxt(f, vertices, fmt='%f %f %f %d %d %d')

def main():
    """
    Ana çalışma fonksiyonu. Kamerayı başlatır, kareleri işler ve sonucu kaydeder.
    """
    resolution_width, resolution_height = (640, 480)
    # Belirtilen metrenin üzerindeki derinlik değerlerini kaldır
    clip_distance_max = 3.5  

    # Kamerayı başlat
    realsense_cam = DepthCamera(resolution_width, resolution_height)
    depth_scale = realsense_cam.get_depth_scale()
    
    # Akış profilinden içsel parametreleri al
    profile = realsense_cam.profile
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    print("Kamera başlatıldı. Canlı görüntü için pencereye bakın.")
    print("'q' tuşuna basarak bir renkli görüntü, bir derinlik görüntüsü ve bir nokta bulutu dosyası kaydedip çıkabilirsiniz.")

    try:
        while True:
            # Ham kareleri al
            ret, depth_raw_frame, color_raw_frame = realsense_cam.get_raw_frame()
            if not ret:
                print("Kare alınamadı, devam ediliyor...")
                continue
            
            # Görüntüleri numpy dizilerine dönüştür
            color_image = np.asanyarray(color_raw_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(depth_raw_frame.get_data()), alpha=0.03), cv2.COLORMAP_JET)

            # Görüntüleri birleştirip göster
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense - Renkli ve Derinlik", images)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' tuşuna basıldığında çık
            if key == ord('q'):
                print("Çıkış yapılıyor ve dosyalar kaydediliyor...")
                
                # Renkli ve derinlik görüntülerini kaydet
                cv2.imwrite("renkli_goruntu.png", color_image)
                plt.imsave("derinlik_goruntu.png", np.asanyarray(depth_raw_frame.get_data()), cmap='viridis')
                print("Renkli ve derinlik görüntüleri kaydedildi.")

                # Nokta bulutunu oluştur ve dosyaya yaz
                points_xyz_rgb = depth2PointCloud(depth_raw_frame, color_raw_frame, intrinsics, depth_scale, clip_distance_max)
                create_point_cloud_file(points_xyz_rgb, "nokta_bulutu.ply")
                print("Nokta bulutu 'nokta_bulutu.ply' olarak kaydedildi.")
                
                break
    finally:
        # Kaynakları serbest bırak
        realsense_cam.release()
        cv2.destroyAllWindows()
        print("Kamera kaynakları serbest bırakıldı.")

if __name__ == '__main__':
    main()