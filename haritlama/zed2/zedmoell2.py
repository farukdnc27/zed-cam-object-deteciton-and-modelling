import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d
import os
from datetime import datetime

# --- 1. YAPILANDIRMA ---
OUTPUT_FOLDER = "final_models"

# --- 2. GLOBAL DEĞİŞKENLER (Dikdörtgen Çizimi İçin) ---
is_drawing = False
start_point = (-1, -1)
end_point = (-1, -1)

# --- 3. YARDIMCI FONKSİYONLAR ---
def validate_points(points):
    """Point cloud verilerini doğrular ve temizler"""
    if points is None or points.size == 0:
        return None
    
    # NaN ve inf değerleri kontrol et
    finite_mask = np.isfinite(points).all(axis=-1)
    valid_points = points[finite_mask]
    
    if valid_points.size == 0:
        return None
    
    # Şekil kontrolü - 3D nokta olmalı
    if len(valid_points.shape) == 3 and valid_points.shape[2] == 4:
        # XYZW formatındaysa sadece XYZ al
        valid_points = valid_points[:, :, :3]
    elif len(valid_points.shape) == 2 and valid_points.shape[1] == 4:
        # 2D array olarak XYZW formatındaysa
        valid_points = valid_points[:, :3]
    
    # Reshape to 2D if needed
    if len(valid_points.shape) == 3:
        valid_points = valid_points.reshape(-1, 3)
    
    # Aşırı uzak noktaları filtrele (10 metre üzeri)
    distances = np.linalg.norm(valid_points, axis=1)
    valid_points = valid_points[distances < 10.0]
    
    return valid_points if valid_points.size > 0 else None

def safe_vstack(point_arrays):
    """Güvenli şekilde point array'leri birleştirir"""
    valid_arrays = []
    for arr in point_arrays:
        validated = validate_points(arr)
        if validated is not None:
            valid_arrays.append(validated)
    
    if not valid_arrays:
        return None
    
    try:
        return np.vstack(valid_arrays)
    except Exception as e:
        print(f"vstack hatası: {e}")
        return None

# --- 4. ANA UYGULAMA ---
def main():
    # --- BAŞLATMA ---
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD2K,
        depth_mode=sl.DEPTH_MODE.QUALITY,
        coordinate_units=sl.UNIT.METER,
        sdk_verbose=1  # Hata ayıklama için verbose açtım
    )
    
    if (err := zed.open(init_params)) != sl.ERROR_CODE.SUCCESS: 
        print(f"HATA: Kamera açılamadı: {err}")
        exit(1)

    zed.enable_positional_tracking()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Modelleme Sahnesi")
    cv2.namedWindow("Kontrol Penceresi (Kamera)")
    
    point_cloud_o3d = o3d.geometry.PointCloud()
    is_pcd_in_vis = False
    
    runtime_params = sl.RuntimeParameters()
    zed_image, point_cloud_zed = sl.Mat(), sl.Mat()
    all_collected_points = []

    def mouse_events(event, x, y, flags, param):
        """Fare olaylarını yönetir ve seçilen alandaki noktaları toplar."""
        global is_drawing, start_point, end_point
        nonlocal all_collected_points

        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
            start_point = (x, y)
            end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing:
                end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False
            end_point = (x, y)
            
            x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
            x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])

            if x1 < x2 and y1 < y2:
                print(f"Alan seçildi: ({x1},{y1}) -> ({x2},{y2}). Noktalar toplanıyor...")
                
                try:
                    # Point cloud verisini al
                    pc_data = point_cloud_zed.get_data()
                    print(f"Point cloud şekli: {pc_data.shape}, veri türü: {pc_data.dtype}")
                    
                    # Seçilen alanı crop et
                    selection_rect = pc_data[y1:y2, x1:x2]
                    print(f"Seçilen alan şekli: {selection_rect.shape}")
                    
                    # Noktaları doğrula ve temizle
                    valid_points = validate_points(selection_rect)
                    
                    if valid_points is not None and valid_points.shape[0] > 0:
                        all_collected_points.append(valid_points)
                        print(f"{valid_points.shape[0]} nokta eklendi. Toplam: {sum(len(p) for p in all_collected_points)}")
                    else:
                        print("Seçilen alanda geçerli 3D nokta bulunamadı.")
                        
                except Exception as e:
                    print(f"Nokta toplama hatası: {e}")
                    print(f"Hata türü: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
    
    cv2.setMouseCallback("Kontrol Penceresi (Kamera)", mouse_events)

    print("\n--- DİKDÖRTGEN SEÇİMİ İLE MODELLEME ---")
    print(" Fare Sol Tuşu (Basılı Tut ve Sürükle): Alan Seç")
    print(" 's': Model Oluştur ve Kaydet")
    print(" 'c': Temizle")
    print(" 'd': Debug bilgisi göster")
    print(" 'q': Çıkış")
    print("-------------------------------------------\n")

    frame_count = 0
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZ)
            cv_image = zed_image.get_data()

            if is_drawing:
                cv2.rectangle(cv_image, start_point, end_point, (0, 255, 0), 2)

            # Görselleştirmeyi güvenli şekilde yap
            if len(all_collected_points) > 0:
                try:
                    combined_points = safe_vstack(all_collected_points)
                    
                    if combined_points is not None and combined_points.shape[0] > 0:
                        # Open3D için veri türünü kontrol et
                        if combined_points.dtype != np.float64:
                            combined_points = combined_points.astype(np.float64)
                        
                        point_cloud_o3d.points = o3d.utility.Vector3dVector(combined_points)
                        
                        if not is_pcd_in_vis:
                            vis.add_geometry(point_cloud_o3d)
                            is_pcd_in_vis = True
                        else:
                            vis.update_geometry(point_cloud_o3d)
                    else:
                        print("Birleştirilen nokta verisi geçersiz")
                        all_collected_points = []
                        
                except Exception as e:
                    print(f"Görselleştirme Hatası: {e}")
                    print(f"Hata türü: {type(e).__name__}")
                    # Hatalı veriyi temizle
                    all_collected_points = []
                    if is_pcd_in_vis:
                        point_cloud_o3d.clear()
                        vis.update_geometry(point_cloud_o3d)
            
            cv2.imshow("Kontrol Penceresi (Kamera)", cv_image)
            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                print("Çıkılıyor...")
                break
            
            elif key == ord('d'):
                # Debug bilgisi
                frame_count += 1
                pc_data = point_cloud_zed.get_data()
                print(f"\n--- DEBUG BİLGİSİ (Frame {frame_count}) ---")
                print(f"Point cloud şekli: {pc_data.shape}")
                print(f"Point cloud veri türü: {pc_data.dtype}")
                print(f"Min değerler: {np.nanmin(pc_data, axis=(0,1))}")
                print(f"Max değerler: {np.nanmax(pc_data, axis=(0,1))}")
                print(f"NaN sayısı: {np.sum(np.isnan(pc_data))}")
                print(f"Inf sayısı: {np.sum(np.isinf(pc_data))}")
                print(f"Toplanan nokta grubu sayısı: {len(all_collected_points)}")
                if all_collected_points:
                    total_points = sum(len(p) for p in all_collected_points)
                    print(f"Toplam nokta sayısı: {total_points}")
                print("-----------------------------------\n")
            
            elif key == ord('s'):
                if len(all_collected_points) > 0:
                    print("\nModel oluşturuluyor...")
                    try:
                        combined_points = safe_vstack(all_collected_points)
                        
                        if combined_points is not None and combined_points.shape[0] > 100:  # En az 100 nokta
                            final_pcd = o3d.geometry.PointCloud()
                            final_pcd.points = o3d.utility.Vector3dVector(combined_points.astype(np.float64))
                            
                            print("Outlier temizleniyor...")
                            final_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                            
                            print("Yüzey oluşturuluyor (Poisson)...")
                            final_pcd.estimate_normals(
                                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                            )
                            
                            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                                final_pcd, depth=8
                            )
                            
                            print("Model kaydediliyor...")
                            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filepath = os.path.join(OUTPUT_FOLDER, f"final_model_{timestamp}.ply")
                            
                            if o3d.io.write_triangle_mesh(filepath, mesh):
                                print(f"BAŞARILI! Model kaydedildi: {os.path.abspath(filepath)}")
                            else:
                                print("HATA: Model kaydedilemedi.")
                        else:
                            print("HATA: Yeterli geçerli nokta bulunamadı (min: 100)")
                    except Exception as e:
                        print(f"Model oluşturma hatası: {e}")
                        import traceback
                        traceback.print_exc()
                else: 
                    print("Kaydedilecek nokta bulunmuyor.")

            elif key == ord('c'):
                print("Temizleniyor...")
                all_collected_points = []
                if is_pcd_in_vis:
                    point_cloud_o3d.clear()
                    vis.update_geometry(point_cloud_o3d)
                
    zed.close()
    vis.destroy_window()
    cv2.destroyAllWindows()
    print("Uygulama kapatıldı.")

if __name__ == "__main__":
    main()