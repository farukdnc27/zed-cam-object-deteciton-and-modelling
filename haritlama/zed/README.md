# ZED Nesne Tespiti ve Hacim Tahmini

Bu proje, **ZED 2i** stereoskopik kamera ve **YOLOv8** nesne tespit modelini birleştirerek, tespit edilen nesnelerin (şişe, kutu, bilgisayar vb.) 3D dünyadaki konumlarını ve hacimlerini hesaplar.

## 🌟 Özellikler

*   **YOLOv8 Entegrasyonu:** Nesneleri gerçek zamanlı tanır.
*   **3D Konumlandırma:** ZED derinlik haritasını kullanarak nesnenin 3D koordinatlarını bulur.
*   **Hacim Hesabı:** Nesnenin türüne göre (silindir veya prizma) hacmini litre/ml cinsinden tahmin eder.
*   **AR Görselleştirme:** Nesnelerin etrafına 3D bounding box (sınırlayıcı kutu) çizer.
*   **Open3D Görünümü:** Sahneyi ve tespit edilen nesneleri 3D uzayda görselleştirir.

## 📂 Önemli Dosyalar

*   **`zed11.py`**: Projenin en güncel ve kapsamlı ana dosyasıdır. Hem OpenCV penceresinde AR çizimi yapar hem de Open3D penceresinde 3D sahneyi gösterir.
*   `yolov8n.pt`: YOLOv8 model dosyası.

## 📦 Gereksinimler

*   ZED SDK ve Python API (`pyzed`)
*   `ultralytics` (YOLOv8 için)
*   `opencv-python`
*   `open3d`
*   `numpy`

```bash
pip install ultralytics opencv-python open3d numpy
```

## ▶️ Kullanım

```bash
python haritlama/zed/zed11.py
```

Program çalıştığında iki pencere açılacaktır:
1.  **ZED - 2D AR Görünümü:** Kamera görüntüsü üzerinde kutular ve hacim bilgileri.
2.  **3D Sahne Gösterimi:** Open3D ile oluşturulan nokta bulutu ve nesne kutuları.
