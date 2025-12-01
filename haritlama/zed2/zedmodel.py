import pyzed.sl as sl
import numpy as np
import cv2

def main():
    # --- 1. ZED Kamerayı Başlatma ---
    print("ZED kamerası başlatılıyor...")
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Kamera açılamadı:", status)
        exit()
    print("ZED kamerası başarıyla açıldı.")

    # --- 2. Pozisyon Takibini Açma ---
    print("Pozisyon takibi başlatılıyor...")
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    # --- 3. Uzamsal Haritalamayı Açma ---
    print("Uzamsal haritalama (Spatial Mapping) başlatılıyor...")
    mapping_params = sl.SpatialMappingParameters()
    mapping_params.save_texture = True
    zed.enable_spatial_mapping(mapping_params)

    # --- 4. Kullanıcı Arayüzü İçin Değişkenler ve Pencereler ---
    print("Taramaya başla... Kamera görüntüsü ekranda.")
    print("Kutuyu yavaşça farklı açılardan tarayın.")
    print("Taramayı bitirmek ve modeli kaydetmek için 'k' tuşuna basın.")
    print("Çıkmak için 'q' tuşuna basın.")
    
    window_name = "ZED 3D Modelleme"
    cv2.namedWindow(window_name)
    image_zed = sl.Mat()
    runtime = sl.RuntimeParameters()

    # --- 5. Tarama Döngüsü ---
    is_mapping = True
    while is_mapping:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Canlı görüntüyü al
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image_ocv = image_zed.get_data()

            # Uzamsal harita verisini eşzamansız olarak iste
            zed.request_spatial_map_async()

            # Tarama ilerlemesini al ve ekrana yaz
            mapping_state = zed.get_spatial_mapping_state()
            
            # ZED SDK 5.0.5'te ilerleme yüzdesini doğrudan almak için
            # get_spatial_mapping_mesh_request_status() yerine get_mesh_fill_state() kullanılır.
            # Ancak bu fonksiyona doğrudan erişim değişmiş olabilir. 
            # En sağlam yol, state'i kontrol etmektir.
            
            # Bu fonksiyon, ZED SDK 5.0.5'te tarama durumunu verir, 
            # ancak yüzdesel ilerleme bilgisi ayrı bir nesne aracılığıyla gelir.
            # Bu nedenle, sadece durumu yazdıralım.
            
            info_text = f"Durum: {mapping_state}"
            cv2.putText(image_ocv, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow(window_name, image_ocv)

        key = cv2.waitKey(1) & 0xFF

        # 'k' tuşuna basıldığında modelleme döngüsünden çık
        if key == ord('k'):
            is_mapping = False
            print("Tarama durduruldu. Mesh oluşturuluyor...")
        # 'q' tuşuna basıldığında tüm programdan çık
        elif key == ord('q'):
            print("Program sonlandırılıyor...")
            is_mapping = False
            zed.close()
            cv2.destroyAllWindows()
            exit()

    # --- 6. Mesh Oluşturma ve Kaydetme ---
    print("Son mesh verisi çıkarılıyor...")
    mesh = sl.Mesh()
    zed.extract_whole_spatial_map(mesh)

    # Mesh'i dosyaya kaydet
    output_path = "3d_model.obj"
    mesh.save(output_path)
    print(f"Mesh başarıyla kaydedildi: {output_path}")

    # --- 7. Temizlik ---
    print("Temizlik yapılıyor...")
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()