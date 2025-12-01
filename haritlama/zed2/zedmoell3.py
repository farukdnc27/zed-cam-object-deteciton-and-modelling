import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d
import os
from datetime import datetime


OUTPUT_FOLDER = "final_models"


is_drawing = False
start_point = (-1, -1)
end_point = (-1, -1)


def advanced_point_validation(points, depth_threshold=(0.3, 5.0)):
    """Gelişmiş nokta doğrulaması ve filtreleme"""
    if points is None or points.size == 0:
        return None
    
    
    if len(points.shape) == 3:
        h, w, c = points.shape
        points_2d = points.reshape(-1, c)
    else:
        points_2d = points.copy()
    
    
    if points_2d.shape[1] == 4:
        points_2d = points_2d[:, :3]
    
    
    finite_mask = np.isfinite(points_2d).all(axis=1)
    valid_points = points_2d[finite_mask]
    
    if valid_points.size == 0:
        return None
    
    
    distances = np.linalg.norm(valid_points, axis=1)
    depth_mask = (distances > depth_threshold[0]) & (distances < depth_threshold[1])
    valid_points = valid_points[depth_mask]
    
    
    if valid_points.size > 0:
        z_values = valid_points[:, 2]
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)
       
        z_mask = np.abs(z_values - z_mean) < 3 * z_std
        valid_points = valid_points[z_mask]
    
   
    if valid_points.shape[0] < 50:
        print(f"Uyarı: Çok az nokta ({valid_points.shape[0]}). Daha geniş alan seçin.")
        return valid_points if valid_points.shape[0] > 10 else None
    
    return valid_points

def create_high_quality_mesh(points, method='poisson'):
    """Yüksek kaliteli mesh oluşturma"""
    if points is None or len(points) < 100:
        return None
    
    print(f"Mesh oluşturuluyor: {len(points)} nokta ile...")
    
   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    print("1. Outlier temizleniyor...")
    
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd, _ = pcd.remove_radius_outlier(nb_points=20, radius=0.1)
    
    remaining_points = len(pcd.points)
    print(f"   Kalan nokta: {remaining_points}")
    
    if remaining_points < 100:
        print("Uyarı: Temizleme sonrası çok az nokta kaldı")
        return None
    
    print("2. Normal vektörler hesaplanıyor...")
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )
    
    
    pcd.orient_normals_consistent_tangent_plane(100)
    
    if method == 'poisson':
        print("3. Poisson yüzey rekonstrüksiyonu...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False
        )
        
        
        if len(densities) > 0:
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.1)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
    
    elif method == 'alpha':
        print("3. Alpha shape mesh...")
        
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        alpha = 2 * avg_dist
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    elif method == 'ball_pivoting':
        print("3. Ball Pivoting Algorithm...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 2 * avg_dist
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )
    
    print("4. Mesh temizleniyor...")
    
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    return mesh

def visualize_model_preview(pcd_points):
    """Model önizlemesi göster"""
    if pcd_points is None or len(pcd_points) < 10:
        return
    
    preview_pcd = o3d.geometry.PointCloud()
    preview_pcd.points = o3d.utility.Vector3dVector(pcd_points)
    
    
    colors = np.zeros_like(pcd_points)
    z_values = pcd_points[:, 2]
    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())
    colors[:, 0] = z_norm  # Kırmızı
    colors[:, 2] = 1 - z_norm  # Mavi
    preview_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    
    vis_preview = o3d.visualization.Visualizer()
    vis_preview.create_window("Model Önizleme", width=600, height=400)
    vis_preview.add_geometry(preview_pcd)
    vis_preview.run()
    vis_preview.destroy_window()


def main():
    
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.ULTRA,  
        coordinate_units=sl.UNIT.METER,
        depth_minimum_distance=0.3,  
        depth_maximum_distance=5.0,   
        sdk_verbose=1
    )
    
    if (err := zed.open(init_params)) != sl.ERROR_CODE.SUCCESS: 
        print(f"HATA: Kamera açılamadı: {err}")
        exit(1)

    
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)
    
    
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Modelleme Sahnesi", width=800, height=600)
    cv2.namedWindow("Kontrol Penceresi (Kamera)")
    
    point_cloud_o3d = o3d.geometry.PointCloud()
    is_pcd_in_vis = False
    
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50  # Güven eşiği
    runtime_params.texture_confidence_threshold = 100
    
    zed_image, point_cloud_zed = sl.Mat(), sl.Mat()
    all_collected_points = []
    current_selection_points = None

    def mouse_events(event, x, y, flags, param):
        global is_drawing, start_point, end_point
        nonlocal all_collected_points, current_selection_points

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

            if abs(x2-x1) > 20 and abs(y2-y1) > 20:  # Minimum seçim boyutu
                print(f"Alan seçildi: ({x1},{y1}) -> ({x2},{y2})")
                
                try:
                    pc_data = point_cloud_zed.get_data()
                    selection_rect = pc_data[y1:y2, x1:x2]
                    
                    
                    valid_points = advanced_point_validation(selection_rect)
                    
                    if valid_points is not None and len(valid_points) > 50:
                        current_selection_points = valid_points
                        all_collected_points.append(valid_points)
                        print(f"✓ {len(valid_points)} kaliteli nokta eklendi.")
                        print(f"  Toplam: {sum(len(p) for p in all_collected_points)} nokta")
                        
                        # Mesafe bilgisi
                        distances = np.linalg.norm(valid_points, axis=1)
                        print(f"  Mesafe aralığı: {distances.min():.2f}m - {distances.max():.2f}m")
                    else:
                        print("✗ Bu alanda yeterli kaliteli nokta bulunamadı")
                        
                except Exception as e:
                    print(f"Seçim hatası: {e}")
            else:
                print("Çok küçük alan. Daha büyük bir alan seçin.")
    
    cv2.setMouseCallback("Kontrol Penceresi (Kamera)", mouse_events)

    print("\n🎯 GELİŞMİŞ 3D MODELLEME")
    print("=" * 40)
    print("📹 KONTROLLER:")
    print("  • Fare: Alan seç (sürükleyerek)")
    print("  • 's': Model oluştur (Poisson)")
    print("  • 'a': Model oluştur (Alpha Shape)")  
    print("  • 'b': Model oluştur (Ball Pivoting)")
    print("  • 'p': Önizleme göster")
    print("  • 'c': Temizle")
    print("  • 'd': Debug bilgisi")
    print("  • 'q': Çıkış")
    print("=" * 40)
    print("💡 İPUCU: Objeye 0.5-2m mesafeden, iyi aydınlatmada çekim yapın")
    print()

    frame_count = 0
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZ)
            cv_image = zed_image.get_data()

            # Seçim dikdörtgeni çiz
            if is_drawing:
                cv2.rectangle(cv_image, start_point, end_point, (0, 255, 0), 2)
                # Alan bilgisi göster
                w, h = abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1])
                cv2.putText(cv_image, f"{w}x{h}", 
                           (start_point[0], start_point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Toplanan bölgeleri göster
            if len(all_collected_points) > 0:
                try:
                    combined_points = np.vstack(all_collected_points)
                    if combined_points.dtype != np.float64:
                        combined_points = combined_points.astype(np.float64)
                    
                    point_cloud_o3d.points = o3d.utility.Vector3dVector(combined_points)
                    
                    # Renklendirme (mesafeye göre)
                    distances = np.linalg.norm(combined_points, axis=1)
                    colors = np.zeros_like(combined_points)
                    dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-6)
                    colors[:, 1] = 1 - dist_norm  # Yeşil (yakın)
                    colors[:, 0] = dist_norm      # Kırmızı (uzak)
                    point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
                    
                    if not is_pcd_in_vis:
                        vis.add_geometry(point_cloud_o3d)
                        is_pcd_in_vis = True
                    else:
                        vis.update_geometry(point_cloud_o3d)
                        
                except Exception as e:
                    print(f"Görselleştirme hatası: {e}")
                    all_collected_points = []
            
            # Bilgi paneli
            info_text = [
                f"Toplanan Nokta: {sum(len(p) for p in all_collected_points)}",
                f"Bölge Sayısı: {len(all_collected_points)}",
                f"Frame: {frame_count}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(cv_image, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Kontrol Penceresi (Kamera)", cv_image)
            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            frame_count += 1
            
            if key == ord('q'): 
                break
                
            elif key == ord('p'):
                # Önizleme göster
                if len(all_collected_points) > 0:
                    combined_points = np.vstack(all_collected_points)
                    visualize_model_preview(combined_points)
                else:
                    print("Önizleme için nokta gerekli")
            
            elif key in [ord('s'), ord('a'), ord('b')]:
                # Model oluştur
                method_map = {'s': 'poisson', 'a': 'alpha', 'b': 'ball_pivoting'}
                method = method_map[chr(key)]
                
                if len(all_collected_points) > 0:
                    print(f"\n🔧 {method.upper()} yöntemi ile model oluşturuluyor...")
                    
                    try:
                        combined_points = np.vstack(all_collected_points)
                        mesh = create_high_quality_mesh(combined_points, method)
                        
                        if mesh is not None and len(mesh.vertices) > 0:
                            print(f"✓ Mesh oluşturuldu: {len(mesh.vertices)} köşe, {len(mesh.triangles)} üçgen")
                            
                            # Kaydetme
                            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # PLY formatında kaydet
                            ply_file = os.path.join(OUTPUT_FOLDER, f"model_{method}_{timestamp}.ply")
                            if o3d.io.write_triangle_mesh(ply_file, mesh):
                                print(f"✓ Model kaydedildi: {os.path.abspath(ply_file)}")
                            
                            # OBJ formatında da kaydet
                            obj_file = os.path.join(OUTPUT_FOLDER, f"model_{method}_{timestamp}.obj")
                            if o3d.io.write_triangle_mesh(obj_file, mesh):
                                print(f"✓ OBJ formatı: {os.path.abspath(obj_file)}")
                            
                            # Modeli görselleştir
                            mesh.paint_uniform_color([0.7, 0.7, 0.7])
                            o3d.visualization.draw_geometries([mesh], 
                                                             window_name="Oluşturulan Model",
                                                             width=800, height=600)
                        else:
                            print("✗ Mesh oluşturulamadı. Daha fazla nokta toplayın.")
                    
                    except Exception as e:
                        print(f"✗ Model oluşturma hatası: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("Model için nokta gerekli")

            elif key == ord('c'):
                print("🧹 Temizleniyor...")
                all_collected_points = []
                current_selection_points = None
                if is_pcd_in_vis:
                    point_cloud_o3d.clear()
                    vis.update_geometry(point_cloud_o3d)
                    
            elif key == ord('d'):
                # Debug bilgisi
                pc_data = point_cloud_zed.get_data()
                print(f"\n🔍 DEBUG BİLGİSİ")
                print(f"Point cloud: {pc_data.shape} {pc_data.dtype}")
                print(f"Toplanan bölge: {len(all_collected_points)}")
                if len(all_collected_points) > 0:
                    total = sum(len(p) for p in all_collected_points)
                    print(f"Toplam nokta: {total}")
                    combined = np.vstack(all_collected_points)
                    distances = np.linalg.norm(combined, axis=1)
                    print(f"Mesafe aralığı: {distances.min():.2f}m - {distances.max():.2f}m")
                print()
                
    zed.close()
    vis.destroy_window()
    cv2.destroyAllWindows()
    print("👋 Uygulama kapatıldı.")

if __name__ == "__main__":
    main()
