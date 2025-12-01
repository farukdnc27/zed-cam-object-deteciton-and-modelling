# Intel RealSense 3D Araçları

Bu klasör, Intel RealSense derinlik kameraları (D415, D435, D455 vb.) kullanılarak geliştirilmiş temel 3D işlem scriptlerini içerir.

## 🛠 İçerik

### 1. 3D Modelleme (`modelleme.py`)
Kameradan alınan derinlik verisini kullanarak anlık 3D nokta bulutu (Point Cloud) oluşturur.
*   **Çıktı:** `nokta_bulutu.ply` dosyası.
*   **Kullanım:** `q` tuşu ile kaydet ve çık.

### 2. 3D Haritalama (`haritalama.py`)
Open3D kütüphanesini kullanarak gerçek zamanlı ortam haritalaması (Reconstruction) yapar. Kamerayı hareket ettirerek odanın 3D modelini çıkarabilirsiniz.
*   **Yöntem:** TSDF Volume Integration.
*   **Çıktı:** `oda_haritasi.ply`
*   **Kullanım:** `CTRL+C` ile bitir.

### 3. Mesafe Ölçümü (`olcum.py`)
Kamera görüntüsü üzerinde tıklanan noktalar arasındaki mesafeyi ölçer.
*   **Özellik:** Çoklu ölçüm desteği.
*   **Kullanım:** Sol tık ile nokta seç, `SPACE` ile temizle.

## 📦 Kurulum

```bash
pip install pyrealsense2 numpy opencv-python open3d matplotlib
```

## ▶️ Çalıştırma

Ana dizinden:
```bash
python pythonzed/modelleme.py
python pythonzed/haritalama.py
python pythonzed/olcum.py
```
