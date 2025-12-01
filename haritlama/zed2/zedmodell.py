import pyzed.sl as sl
from datetime import datetime
import os
import sys # sys modülünü ekliyoruz

def main():
    # Kamera başlatma
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Kamera açılamadı: {err}")
        exit(1)

    # Positional Tracking'i başlat
    tracking_params = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Positional Tracking başlatılamadı: {err}")
        zed.close()
        exit(1)

    # Spatial Mapping ayarları
    mapping_params = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.MEDIUM,
        mapping_range=sl.MAPPING_RANGE.MEDIUM,
        save_texture=True
    )
    err = zed.enable_spatial_mapping(mapping_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Spatial Mapping başlatılamadı: {err}")
        zed.close()
        exit(1)

    # Tarama döngüsü
    runtime_params = sl.RuntimeParameters()
    scan_duration = 15
    start_time = datetime.now()

    print("Tarama başladı... Kamerayı yavaşça ve sabit bir şekilde hareket ettirin.")
    while (datetime.now() - start_time).seconds < scan_duration:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            pass

    print("\nTarama tamamlandı. Mesh oluşturuluyor...")
    
    mesh = sl.Mesh()
    zed.extract_whole_spatial_map(mesh)
    print(f"Ham mesh oluşturuldu: {mesh.vertices.shape[0]} vertex, {mesh.triangles.shape[0]} üçgen")

    # Filtrelemeyi şimdilik kapalı tutuyoruz, en azından bir model elde etmek için.
    # print("Daha temiz bir model için mesh filtreleniyor...")
    # mesh.filter(sl.MESH_FILTER.LOW) # Gerekirse daha sonra LOW ile deneyebilirsiniz.
    # print(f"Filtrelenmiş mesh: {mesh.vertices.shape[0]} vertex, {mesh.triangles.shape[0]} üçgen")

    # Mesh'i kaydet
    if mesh.vertices.shape[0] > 0:
        # --- NİHAİ ÇÖZÜM: PROJE DİZİNİNE GÖRELİ KAYDETME ---
        
        # 1. Betiğin çalıştığı dizinde "zed_output" adında bir klasör oluştur.
        #    exist_ok=True sayesinde klasör zaten varsa hata vermez.
        output_dir = "zed_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Model dosyasının tam yolunu oluştur. Örn: zed_output/object_model.ply
        output_filename = "object_model.ply"
        output_path = os.path.join(output_dir, output_filename)
        
        # 3. Bu yolu kullanarak modeli kaydet.
        print(f"\nModel proje dizinindeki '{output_dir}' klasörüne kaydediliyor...")
        status = mesh.save(output_path, sl.MESH_FILE_FORMAT.PLY)
        
        if status:
            # Kaydedilen dosyanın tam yolunu kullanıcıya göstermek için
            full_path = os.path.abspath(output_path)
            print(f"\nBaşarılı! Model kaydedildi:\n{full_path}")
        else:
            print(f"\nHATA: Model kaydedilemedi. ZED SDK Durum Kodu: {status}")
            print("İPUCU: Lütfen komut istemini (PowerShell/CMD) 'Yönetici Olarak Çalıştır' seçeneği ile açıp betiği tekrar deneyin.")

    else:
        print("\nHATA: Boş mesh oluşturuldu. Filtreleme çok agresif olabilir veya tarama başarısız oldu.")

    # Kaynakları serbest bırak
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    print("\nİşlem tamamlandı. Kamera kapatıldı.")

if __name__ == "__main__":
    main()