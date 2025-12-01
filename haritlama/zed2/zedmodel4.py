import pyzed.sl as sl
import numpy as np
import cv2
import os
from datetime import datetime

class ZED3DScanner:
    def __init__(self):
        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.init_params = sl.InitParameters()
        self.configure_camera()
        
    def configure_camera(self):
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.camera_fps = 30
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Kamera açılamadı: {err}")
            exit(1)
        print("ZED kamerası başarıyla başlatıldı")

    def scan_3d_model(self):
        # Spatial mapping parametreleri
        mapping_params = sl.SpatialMappingParameters()
        mapping_params.resolution_meter = 0.02  # Orta çözünürlük
        mapping_params.save_texture = True  # Doku kaydını etkinleştir
        mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH
        mapping_params.use_chunk_only = False

        # Pozisyon takibi
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.set_as_static = True
        self.zed.enable_positional_tracking(tracking_params)
        
        # Spatial mapping'i başlat
        err = self.zed.enable_spatial_mapping(mapping_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"3D mapping başlatılamadı: {err}")
            return False

        print("\n3D Tarama Başladı...")
        print("Kamerayı nesnenin etrafında yavaşça hareket ettirin")
        print("Tarama tamamlandığında 's' tuşuna basarak kaydedin")
        print("İptal için 'q' tuşuna basın")

        image = sl.Mat()
        window_name = "3D Tarama - ZED"
        cv2.namedWindow(window_name)

        scanning = True
        while scanning:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Görüntüyü al ve göster
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                cv_image = image.get_data()
                
                # Tarama durumunu göster
                state = self.zed.get_spatial_mapping_state()
                cv2.putText(cv_image, f"Durum: {state.name}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow(window_name, cv_image)
                
                key = cv2.waitKey(10)
                if key == ord('s'):  # Kaydet
                    scanning = False
                elif key == ord('q'):  # İptal
                    cv2.destroyAllWindows()
                    self.zed.disable_spatial_mapping()
                    return False

        cv2.destroyAllWindows()
        
        # Mesh'i çıkar
        print("\nMesh oluşturuluyor...")
        mesh = sl.Mesh()
        filter_params = sl.MeshFilterParameters()
        filter_params.set(sl.MESH_FILTER.MEDIUM)
        
        # Tüm mesh'i çıkar
        status = self.zed.extract_whole_spatial_map(mesh)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Mesh çıkarılamadı: {status}")
            return False
        
        # Mesh'i filtrele
        print("Mesh filtreleniyor...")
        mesh.filter(filter_params, True)
        
        # Kaydet
        output_dir = "zed_3d_models"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(output_dir, f"model_{timestamp}")
        obj_filename = base_filename + ".obj"
        
        print(f"\nModel kaydediliyor: {obj_filename}")
        
        # Texture'ı ayrıca kaydet
        if mesh.textures.is_init():
            texture_filename = base_filename + ".png"
            print(f"Doku kaydediliyor: {texture_filename}")
            texture_image = sl.Mat()
            texture_image.set(mesh.textures)
            cv2.imwrite(texture_filename, texture_image.get_data())
        
        # Mesh'i kaydet
        save_params = sl.MESH_FILE_FORMAT.OBJ
        print("Kayıt parametreleri ayarlanıyor...")
        
        # Mesh'i kaydet
        print("Mesh dosyaya yazılıyor...")
        save_status = mesh.save(obj_filename, save_params)
        
        if save_status:
            print("3D model başarıyla kaydedildi!")
            print(f"Üçgen sayısı: {mesh.triangles.size}")
            print(f"Köşe sayısı: {mesh.vertices.size}")
            
            # Kaydedilen dosyaları kontrol et
            created_files = []
            if os.path.exists(obj_filename):
                print(f"OBJ dosyası oluşturuldu: {os.path.getsize(obj_filename)} bytes")
                created_files.append(obj_filename)
                
                mtl_file = base_filename + ".mtl"
                if os.path.exists(mtl_file):
                    print(f"MTL dosyası oluşturuldu: {os.path.getsize(mtl_file)} bytes")
                    created_files.append(mtl_file)
                else:
                    print("MTL dosyası oluşturulamadı!")
                
                if mesh.textures.is_init():
                    texture_file = base_filename + ".png"
                    if os.path.exists(texture_file):
                        print(f"Doku dosyası oluşturuldu: {os.path.getsize(texture_file)} bytes")
                        created_files.append(texture_file)
                    else:
                        print("Doku dosyası oluşturulamadı!")
                
                return True
            else:
                print("OBJ dosyası oluşturulamadı!")
                return False
        else:
            print("Model kaydedilemedi!")
            return False

    def close(self):
        self.zed.disable_spatial_mapping()
        self.zed.disable_positional_tracking()
        self.zed.close()
        cv2.destroyAllWindows()

def main():
    scanner = ZED3DScanner()
    try:
        success = scanner.scan_3d_model()
        if success:
            print("\n3D tarama ve kaydetme işlemi başarıyla tamamlandı!")
        else:
            print("\n3D tarama işlemi başarısız oldu")
    except Exception as e:
        print(f"\nBir hata oluştu: {str(e)}")
    finally:
        scanner.close()

if __name__ == "__main__":
    main()