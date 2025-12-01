import pyzed.sl as sl
import numpy as np
import cv2

# Global değişkenler
selected_area_2d = None
is_drawing = False
start_point = None
current_point = None # Yeni eklenen değişken

# Fare etkinliği callback fonksiyonu
def mouse_callback(event, x, y, flags, param):
    global selected_area_2d, is_drawing, start_point, current_point

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        start_point = (x, y)
        current_point = (x, y) # Tıklama anında da current_point'i ayarla

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            current_point = (x, y) # Fare sürüklenirken mevcut konumu güncelle

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        end_point = (x, y)
        selected_area_2d = (start_point, end_point)
        print("Seçili alan: ", selected_area_2d)

def main():
    global selected_area_2d, start_point, current_point

    # 1. Init Parametreler
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER

    # 2. Kamera Aç
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Kamera açılamadı:", status)
        exit()

    # 3. Görüntüleri almak için mat nesneleri
    image_zed = sl.Mat()
    point_cloud_zed = sl.Mat()

    # 4. OpenCV penceresini oluştur
    window_name = "ZED Goruntu Akisi"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Lütfen fare ile taramak istediğiniz nesnenin etrafına bir dikdörtgen çizin.")
    
    # Kullanıcı bir alan seçene kadar bekle
    while selected_area_2d is None:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image_ocv = image_zed.get_data()
            
            # Görüntü üzerine anlık çizim yap
            if is_drawing and start_point is not None and current_point is not None:
                # Burada 'current_point' değişkenini kullanıyoruz
                cv2.rectangle(image_ocv, start_point, current_point, (255, 0, 0), 2)
            
            cv2.imshow(window_name, image_ocv)
            
            # Kullanıcının 'q' tuşuyla çıkmasını sağla
            if cv2.waitKey(1) & 0xFF == ord('q'):
                zed.close()
                cv2.destroyAllWindows()
                return

    # 5. ... (Diğer kodlar aynı kalabilir)
    # ...
    # ...
    # ...
    
    cv2.destroyAllWindows()

    # 6. Positional tracking başlat
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    # 7. Spatial mapping'i başlat
    mapping_params = sl.SpatialMappingParameters()
    mapping_params.save_texture = True
    zed.enable_spatial_mapping(mapping_params)
    
    print("Tarama başladı. Lütfen sadece belirlediğiniz alanı yavaşça tarayın...")

    # 8. Tarama işlemi (örnek: 200 kare)
    runtime = sl.RuntimeParameters()
    for _ in range(200):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.request_spatial_map_async()

    print("Tarama tamamlandı, mesh işleniyor...")

    # 9. Mesh oluştur
    mesh = sl.Mesh()
    zed.extract_whole_spatial_map(mesh)

    # 10. Mesh'i dosyaya kaydet
    output_path = "belirlenen_alan_modeli.obj"
    mesh.save(output_path)
    print(f"Mesh başarıyla kaydedildi: {output_path}")

    # 11. Temizlik
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()


if __name__ == "__main__":
    main()