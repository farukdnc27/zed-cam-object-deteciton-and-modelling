import pyzed.sl as sl
import sys

def create_box_obj(filename, dimensions):
    """
    Verilen boyutlarla bir kutu modeli oluşturan ve .obj dosyası olarak kaydeden fonksiyon.
    
    :param filename: Kaydedilecek dosyanın adı (örn: "kutu.obj")
    :param dimensions: [genislik, yukseklik, uzunluk] formatında bir liste/tuple
    """
    width, height, depth = dimensions[0], dimensions[1], dimensions[2]
    
    # Boyutların geçerli olup olmadığını kontrol et
    if width <= 0 or height <= 0 or depth <= 0:
        print(f"Hata: Geçersiz boyutlar! {dimensions}. Model oluşturulamadı.")
        return

    # Kutunun 8 köşesinin koordinatlarını hesapla
    # Orijin (0,0,0) kutunun merkezinde olacak şekilde
    w2, h2, d2 = width / 2.0, height / 2.0, depth / 2.0
    
    vertices = [
        (-w2, -h2,  d2), ( w2, -h2,  d2), ( w2,  h2,  d2), (-w2,  h2,  d2), # Ön yüz
        (-w2, -h2, -d2), ( w2, -h2, -d2), ( w2,  h2, -d2), (-w2,  h2, -d2)  # Arka yüz
    ]

    # .obj formatında yüzler, köşe indekslerini kullanarak tanımlanır.
    # Dikkat: .obj formatında indeksler 1'den başlar!
    faces = [
        (1, 2, 3, 4), # Ön yüz
        (6, 5, 8, 7), # Arka yüz
        (1, 5, 6, 2), # Alt yüz
        (4, 3, 7, 8), # Üst yüz
        (1, 4, 8, 5), # Sol yüz
        (2, 6, 7, 3)  # Sağ yüz
    ]

    try:
        with open(filename, 'w') as f:
            f.write("# ZED 2i ile ölçülen nesne\n")
            f.write(f"# Boyutlar (G,Y,U): {width:.2f} x {height:.2f} x {depth:.2f} mm\n\n")

            # Köşeleri dosyaya yaz
            for v in vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
            f.write("\n")
            
            # Yüzleri dosyaya yaz
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")
        print(f"Model '{filename}' adıyla başarıyla oluşturuldu!")
    except IOError as e:
        print(f"Dosya yazılamadı: {e}")

def main():
    # 1. Kamera ve parametreleri oluştur
    zed = sl.Camera()
    init_params = sl.InitParameters()
    
    # Eski SDK sürümleri NEURAL modunu desteklemeyebilir.
    # Hata alırsanız, aşağıdaki satırı sl.DEPTH_MODE.PERFORMANCE ile değiştirin.
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # 2. Kamerayı aç
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Kamera açılamadı: {err}")
        print("İpucu: Eğer ZED SDK sürümünüz eskiyse, init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE olarak değiştirmeyi deneyin.")
        zed.close()
        return

    # 3. Nesne tespiti modülünü etkinleştir
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    
    # *** UYUMLULUK DÜZELTMESİ ***
    # Bu satır kaldırıldı çünkü eski ZED SDK sürümlerinde `DETECTION_MODEL` bulunmuyor.
    # SDK, varsayılan nesne tespit modelini otomatik olarak kullanacaktır.
    # obj_param.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE 
    
    # Pozisyon takibini etkinleştir (nesne takibi için gerekli)
    if obj_param.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        zed.enable_positional_tracking(positional_tracking_param)

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Nesne tespiti etkinleştirilemedi: {err}")
        zed.close()
        return

    print("Kamera çalışıyor. Nesne tespiti için kamerayı nesneye doğrultun...")
    print("Bir nesne bulunduğunda boyutları yazdırılacak ve program kapanacaktır.")
    print("Çıkmak için Ctrl+C tuşlarına basın.")

    # 4. Veri toplamak için bir nesneler (Objects) nesnesi oluştur
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Nesne bulana kadar döngü
    found_object_dimensions = None
    try:
        while found_object_dimensions is None:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Tespit edilen nesneleri al
                zed.retrieve_objects(objects, obj_runtime_param)
                
                if objects.is_new:
                    object_list = objects.object_list
                    if len(object_list) > 0:
                        first_object = object_list[0]
                        dimensions = first_object.dimensions
                        
                        print("\n--- NESNE BULUNDU! ---")
                        print(f"Nesne ID: {first_object.id}")
                        print(f"Nesne Sınıfı: {str(first_object.label)}") 
                        print(f"Tespit Güveni: {first_object.confidence:.2f}")
                        
                        print(f"\n>>> Boyutlar (Genişlik x Yükseklik x Uzunluk):")
                        print(f">>> {dimensions[0]:.2f} mm x {dimensions[1]:.2f} mm x {dimensions[2]:.2f} mm")
                        
                        found_object_dimensions = dimensions
                        # Döngüyü kırmak için bulunan boyutları değişkene atadık
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından çıkış yapıldı.")

    # 5. Her şeyi kapat ve temizle
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
    
    # Eğer bir boyut bulduysak, şimdi model oluşturma adımına geçebiliriz
    if found_object_dimensions:
        print("\nBu boyutlarla 3D model oluşturuluyor...")
        create_box_obj("olculen_nesne.obj", found_object_dimensions)

# Script'in ana giriş noktası
if __name__ == "__main__":
    main()