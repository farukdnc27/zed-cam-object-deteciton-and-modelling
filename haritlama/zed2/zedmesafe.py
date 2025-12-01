import pyzed.sl as sl
import numpy as np
import cv2

# Global değişkenler
points_to_measure = []
measurements = []
window_name = "ZED Mesafe Olcumu"

# Fare etkinliği callback fonksiyonu
def mouse_callback(event, x, y, flags, param):
    global points_to_measure, measurements

    if event == cv2.EVENT_LBUTTONDOWN:
        # Tıklanan noktanın 2D koordinatını kaydet
        points_to_measure.append((x, y))
        print(f"Nokta secildi: ({x}, {y})")

def main():
    global points_to_measure, measurements, window_name

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

    # 4. OpenCV penceresini oluştur ve fare callback'ini ayarla
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Lütfen ölçüm yapmak için iki nokta seçin.")
    print("Seçimi sıfırlamak için 'r' tuşuna, çıkmak için 'q' tuşuna basın.")

    # 5. Ana döngü
    runtime_params = sl.RuntimeParameters()

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image_ocv = image_zed.get_data()

            # Geçici çizim için görüntünün bir kopyasını al
            display_image = image_ocv.copy()

            # Seçilen tüm noktaları çiz
            for point in points_to_measure:
                cv2.circle(display_image, point, 5, (0, 255, 0), -1)

            # Geçici çizimler: Henüz tamamlanmamış bir ölçüm varsa
            if len(points_to_measure) % 2 == 1 and len(points_to_measure) > 0:
                cv2.circle(display_image, points_to_measure[-1], 5, (0, 0, 255), -1)

            # Kaydedilmiş tüm ölçümleri çiz
            for p1_2d, p2_2d, distance_text in measurements:
                cv2.line(display_image, p1_2d, p2_2d, (255, 255, 255), 2)
                mid_point = ((p1_2d[0] + p2_2d[0]) // 2, (p1_2d[1] + p2_2d[1]) // 2)
                cv2.putText(display_image, distance_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # İki nokta seçildiyse yeni mesafeyi hesapla
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
                        distance = 100 * np.linalg.norm(p1_3d - p2_3d)
                        distance_text = f"{distance:.3f} cm"
                        
                        # Ölçümü kaydedilmiş listeye ekle
                        measurements.append((p1_2d, p2_2d, distance_text))
                        print(f"Mesafe: {distance_text}")

                # Hesaplama sonrası noktaları sıfırla
                points_to_measure = []

            cv2.imshow(window_name, display_image)

            key = cv2.waitKey(1) & 0xFF

            # 'q' tuşuna basıldığında çıkış yap
            if key == ord('q'):
                break
            # 'r' tuşuna basıldığında seçimi ve ölçümleri sıfırla
            elif key == ord('r'):
                points_to_measure = []
                measurements = []
                print("Secimler ve olcumler sıfırlandı.")

    # 6. Temizlik
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()