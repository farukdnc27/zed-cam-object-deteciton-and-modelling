import pyzed.sl as sl


def main():
    # 1. Init Parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.coordinate_units = sl.UNIT.METER  # Ölçekli çalışmak için

    # 2. Kamera Aç
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Kamera açılamadı:", status)
        exit()

    # 3. Alanı taramak için positional tracking başlat
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    # 4. Mesh toplamak için spatial mapping aç
    mapping_params = sl.SpatialMappingParameters()
    mapping_params.save_texture = True
    zed.enable_spatial_mapping(mapping_params)

    print("Taramaya başladım... Lütfen kutuyu yavaşça farklı açılardan göster...")

    # 5. Tarama işlemi (örnek: 200 kare)
    runtime = sl.RuntimeParameters()
    for i in range(200):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.request_spatial_map_async()

    print("Tarama tamamlandı, mesh işleniyor...")

    # 6. Mesh oluştur
    mesh = sl.Mesh()
    zed.extract_whole_spatial_map(mesh)

    # 7. Mesh'i dosyaya kaydet
    output_path = "kutu_modeli.obj"
    mesh.save(output_path)
    print(f"Mesh başarıyla kaydedildi: {output_path}")

    # 8. Temizlik
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()


if __name__ == "__main__":
    main()