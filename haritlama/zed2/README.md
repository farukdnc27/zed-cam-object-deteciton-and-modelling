# ZED Temel Modelleme ve Ölçüm Araçları

Bu klasör, ZED kamerası için geliştirilmiş, kullanımı basit temel araçları içerir. Özellikle hızlı ölçüm almak ve ortamı taramak için tasarlanmıştır.

## 🛠 Araçlar

### 1. Çok Fonksiyonlu Araç (`zedmodel2.py`)
Bu script, menü tabanlı bir arayüz sunar ve iki modda çalışır:

*   **Mod 1: Mesafe Ölçümü**
    *   Ekranda tıkladığınız iki nokta arasındaki gerçek mesafeyi (cm cinsinden) ölçer.
    *   ZED'in derinlik algısını kullanır.
    *   `r`: Sıfırla, `m`: Menüye dön.

*   **Mod 2: 3D Modelleme (Spatial Mapping)**
    *   ZED'in "Spatial Mapping" özelliğini kullanarak ortamın 3D modelini (mesh) çıkarır.
    *   Kamerayı nesne etrafında dolaştırarak tarama yapabilirsiniz.
    *   `k`: Taramayı bitir ve `3d_model.obj` olarak kaydet.

## 📂 Diğer Dosyalar
*   `zedolcum.py`: Sadece ölçüm odaklı script.
*   `*.obj`: Oluşturulan örnek 3D modeller.

## 📦 Gereksinimler

*   ZED SDK ve Python API (`pyzed`)
*   `opencv-python`
*   `numpy`

## ▶️ Kullanım

```bash
python haritlama/zed2/zedmodel2.py
```
Program başladığında terminal üzerinden **1** veya **2** tuşuna basarak modu seçin.
