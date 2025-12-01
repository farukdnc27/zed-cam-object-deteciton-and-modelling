import open3d as o3d

def visualize_point_cloud(file_path):
    """
    Verilen yoldaki bir .ply dosyasını okur ve görselleştirir.
    """
    print(f"'{file_path}' yükleniyor...")
    try:
        # Nokta bulutunu dosyadan oku
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"Hata: Dosya okunamadı. {e}")
        return

    if not pcd.has_points():
        print("Hata: Yüklenen nokta bulutunda hiç nokta bulunmuyor.")
        return

    print("Nokta bulutu başarıyla yüklendi.")
    print("Görselleştirme penceresi açılıyor. Kapatmak için 'q' veya ESC tuşuna basın.")
    
    # Nokta bulutunu görselleştir
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Open3D Nokta Bulutu Görselleştiricisi",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50)

if __name__ == "__main__":
    # Görselleştirmek istediğiniz dosyanın adını buraya yazın
    ply_file = r"C:\Users\faruk\OneDrive\Masaüstü\pythonzed\nokta_bulutu.ply"
    visualize_point_cloud(ply_file)