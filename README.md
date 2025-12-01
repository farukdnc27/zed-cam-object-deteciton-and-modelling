# ZED & RealSense 3D Vision Projects

Bu depo, **Intel RealSense** ve **Stereolabs ZED** derinlik kameraları kullanılarak geliştirilmiş çeşitli bilgisayarlı görü (computer vision), 3D modelleme ve ölçüm projelerini içerir.

Proje, kullanılan donanım ve amaca göre 4 ana bölüme ayrılmıştır:

## 📂 Proje Listesi

### 1. [Intel RealSense Araçları (`pythonzed/`)](./pythonzed)
Intel RealSense kameraları (D400 serisi) için temel araçlar.
*   **Özellikler:** 3D Nokta bulutu oluşturma, gerçek zamanlı oda haritalama (TSDF), mesafe ölçümü.
*   **Ana Dosyalar:** `modelleme.py`, `haritalama.py`, `olcum.py`

### 2. [ZED Nesne Tespiti ve Hacim (`haritlama/zed/`)](./haritlama/zed)
ZED 2i kamera ve YOLOv8 kullanarak nesnelerin tespit edilmesi ve 3D hacimlerinin hesaplanması.
*   **Özellikler:** YOLOv8 entegrasyonu, AR kutu çizimi, nesne hacim tahmini (şişe, kutu vb.).
*   **Ana Dosya:** `zed11.py`

### 3. [ZED Temel Modelleme ve Ölçüm (`haritlama/zed2/`)](./haritlama/zed2)
ZED kamerası ile ortam tarama ve basit ölçüm işlemleri.
*   **Özellikler:** 3D Mesh oluşturma (Spatial Mapping), iki nokta arası mesafe ölçümü.
*   **Ana Dosya:** `zedmodel2.py`

### 4. [ZED Endüstriyel Taş Analizi (`haritlama/zed3/`)](./haritlama/zed3)
Doğal taşların boyutlarının analizi ve kesim planlaması için gelişmiş bir endüstriyel uygulama.
*   **Özellikler:** Voksel tabanlı analiz, paketleme algoritması, kesim planı (CSV) çıktısı, iç zarf (envelope) hesaplama.
*   **Ana Dosya:** `zedtasolcum12.py`

---

## 🚀 Kurulum

Tüm projeler için genel gereksinimler:

```bash
pip install numpy opencv-python open3d
```

**Kamera SDK'ları:**
*   **Intel RealSense:** `pip install pyrealsense2`
*   **ZED Camera:** [ZED SDK](https://www.stereolabs.com/developers/release/) kurulmalı ve Python API'si (`pyzed`) aktif edilmelidir.
*   **YOLO (Sadece `zed` klasörü için):** `pip install ultralytics`

## ⚠️ Notlar
*   `haritlama` klasör ismi projede bu şekilde geçmektedir (haritalama yerine).
*   Her klasörün içinde o projeye özel detaylı `README.md` dosyaları bulunmaktadır.
