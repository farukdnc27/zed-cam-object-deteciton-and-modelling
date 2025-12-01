import pyzed.sl as sl
import numpy as np
import cv2

# --- Global Değişkenler ---
points_to_measure = []
measurements = []

# --- Fare etkinliği callback fonksiyonu (Mesafe Ölçümü) ---
def mouse_callback(event, x, y, flags, param):
    global points_to_measure, measurements
    if event == cv2.EVENT_LBUTTONDOWN:
        points_to_measure.append((x, y))
        print(f"Nokta secildi: ({x}, {y})")

# --- Mod 1: Mesafe Ölçümü ---
def measure_distance_mode(zed):
    global points_to_measure, measurements
    window_name = "ZED Mesafe Olcumu"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Mesafe olcumu modu baslatildi.")
    print("Lütfen ölçüm yapmak için iki nokta seçin.")
    print("Seçimi sıfırlamak için 'r' tuşuna, ana menüye dönmek için 'm' tuşuna basın.")
    
    image_zed = sl.Mat()
    point_cloud_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image_ocv = image_zed.get_data()
            display_image = image_ocv.copy()
            for point in points_to_measure:
                cv2.circle(display_image, point, 5, (0, 255, 0), -1)
            if len(points_to_measure) % 2 == 1 and len(points_to_measure) > 0:
                cv2.circle(display_image, points_to_measure[-1], 5, (0, 0, 255), -1)
            for p1_2d, p2_2d, distance_text in measurements:
                cv2.line(display_image, p1_2d, p2_2d, (255, 255, 255), 2)
                mid_point = ((p1_2d[0] + p2_2d[0]) // 2, (p1_2d[1] + p2_2d[1]) // 2)
                cv2.putText(display_image, distance_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if len(points_to_measure) % 2 == 0 and len(points_to_measure) > 0:
                p1_2d = points_to_measure[-2]
                p2_2d = points_to_measure[-1]
                zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZRGBA)
                err1, p1_3d_data = point_cloud_zed.get_value(p1_2d[0], p1_2d[1])
                err2, p2_3d_data = point_cloud_zed.get_value(p2_2d[0], p2_2d[1])
                if err1 == sl.ERROR_CODE.SUCCESS and err2 == sl.ERROR_CODE.SUCCESS:
                    p1_3d = np.array([p1_3d_data[0], p1_3d_data[1], p1_3d_data[2]])
                    p2_3d = np.array([p2_3d_data[0], p2_3d_data[1], p2_3d_data[2]])
                    if not np.any(np.isnan(p1_3d)) and not np.any(np.isnan(p2_3d)):
                        distance_meters = np.linalg.norm(p1_3d - p2_3d)
                        distance_cm = distance_meters * 100
                        distance_text = f"{distance_cm:.2f} cm"
                        measurements.append((p1_2d, p2_2d, distance_text))
                        print(f"Mesafe: {distance_text}")
                points_to_measure = []
            cv2.imshow(window_name, display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            points_to_measure = []
            measurements = []
            print("Secimler ve olcumler sıfırlandı.")
        elif key == ord('m'):
            cv2.destroyAllWindows()
            print("Ana menüye dönülüyor...")
            return

# --- Mod 2: 3D Modelleme ---
def modeling_mode(zed):
    print("3D Modelleme modu baslatildi.")
    print("Kutuyu yavaşça farklı açılardan tarayın.")
    print("Taramayı bitirmek ve modeli kaydetmek için 'k' tuşuna basın.")
    print("Ana menüye dönmek için 'm' tuşuna basın.")
    mapping_params = sl.SpatialMappingParameters()
    mapping_params.resolution_meter = 0.02
    mapping_params.save_texture = True
    zed.enable_spatial_mapping(mapping_params)

    window_name = "ZED 3D Modelleme"
    cv2.namedWindow(window_name)
    image_zed = sl.Mat()
    runtime = sl.RuntimeParameters()

    is_mapping = True
    while is_mapping:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image_ocv = image_zed.get_data()
            zed.request_spatial_map_async()
            mapping_state = zed.get_spatial_mapping_state()
            info_text = f"Durum: {mapping_state}"
            cv2.putText(image_ocv, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, image_ocv)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            is_mapping = False
            print("Tarama durduruldu. Mesh oluşturuluyor...")
        elif key == ord('m'):
            print("Ana menüye dönülüyor...")
            zed.disable_spatial_mapping()
            cv2.destroyAllWindows()
            return
    
    print("Son mesh verisi çıkarılıyor...")
    mesh = sl.Mesh()
    status_mesh = zed.extract_whole_spatial_map(mesh)

    if status_mesh == sl.ERROR_CODE.SUCCESS:
        output_path = "3d_model.obj"
        print("OBJ, MTL ve doku dosyaları kaydediliyor...")
        
        # HATA DÜZELTME: mesh.save() fonksiyonunun True/False döndürmesi durumu için kontrol
        save_result = mesh.save(output_path, sl.MESH_FILE_FORMAT.OBJ)
        
        if save_result: # Eğer True döndürürse (başarılıysa)
            print(f"Mesh başarıyla kaydedildi: {output_path}")
        else: # Eğer False döndürürse (başarısızsa)
            print("Hata: Mesh kaydedilemedi. Doku verisi yetersiz olabilir veya dosya izni sorunu olabilir.")
    else:
        print(f"Hata: Mesh çıkarılamadı. Hata kodu: {status_mesh}")
    
    zed.disable_spatial_mapping()
    cv2.destroyAllWindows()

# --- Ana Fonksiyon ---
def main():
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.camera_fps = 30
    
    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Kamera açılamadı.")
        exit()
    print("ZED kamerası başarıyla açıldı.")

    while True:
        print("\n--- Ana Menü ---")
        print("1. Mesafe Ölçümü")
        print("2. 3D Modelleme")
        print("q. Çıkış")
        choice = input("Lütfen bir secim yapin: ")
        if choice == '1':
            zed.disable_spatial_mapping()
            zed.disable_positional_tracking()
            zed.enable_positional_tracking(sl.PositionalTrackingParameters())
            measure_distance_mode(zed)
        elif choice == '2':
            zed.disable_spatial_mapping()
            zed.disable_positional_tracking()
            zed.enable_positional_tracking(sl.PositionalTrackingParameters())
            modeling_mode(zed)
        elif choice == 'q':
            break
        else:
            print("Geçersiz secim. Lütfen tekrar deneyin.")

    print("Program sonlandırılıyor...")
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()