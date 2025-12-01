# ZED Endüstriyel Taş Analizi ve Kesim Planlaması

Bu proje, doğal taş bloklarının (veya benzeri düzensiz nesnelerin) ZED kamera ile taranarak boyutlarının analiz edilmesi ve en verimli kesim planının oluşturulması için geliştirilmiştir.

## 🚀 Özellikler

*   **Gelişmiş Voksel Analizi:** Taşın 3D modelini voksel ızgarasına (voxel grid) dönüştürür.
*   **İç Zarf (Inner Envelope):** Taşın pürüzlü yüzeyinden içeri girerek (erosion), taşın içindeki "temiz" ve kullanılabilir hacmi hesaplar.
*   **Paketleme Algoritması (Packing):** Belirlenen hedef boyutlardaki (örn. 30x20x10 cm) kutuların, taşın iç hacmine en verimli şekilde nasıl yerleştirileceğini hesaplar.
*   **Kesim Planı Çıktısı:** Yerleştirilen kutuların koordinatlarını CSV formatında dışa aktarır.
*   **Kalibrasyon:** Referans bir uzunluk kullanarak ölçüm hassasiyetini artırma imkanı.

## 📂 Ana Dosya: `zedtasolcum12.py`

Bu dosya projenin en gelişmiş versiyonudur.

### Klavye Kısayolları (Arayüzde)
*   **C:** Görüntüyü dondur ve analizi başlat (Capture).
*   **R:** Analizi sıfırla, canlı moda dön (Reset).
*   **F:** Kalibrasyon yap (iki noktaya tıkla, gerçek mesafeyi gir).
*   **E:** Sonucu CSV olarak kaydet (Export).
*   **B:** İç zarf (kırmızı kutu) görünümünü aç/kapat.
*   **P:** Yerleştirilen kutuları (yeşil) aç/kapat.
*   **M:** Taşın ham modelini (gri) aç/kapat.
*   **Q:** Çıkış.

## ⚙️ Yapılandırma

Script içinde `StoneDimensionEstimator` sınıfı başlatılırken şu parametreler ayarlanabilir:
*   `voxel_size_mm`: Analiz hassasiyeti (örn. 20mm).
*   `target_dims_cm`: Kesilecek hedef parçaların boyutu.
*   `offset_voxels_inside`: Yüzeyden kaç voksel içeri girileceği (güvenlik payı).

## 📦 Gereksinimler

*   ZED SDK (`pyzed`)
*   `open3d`
*   `opencv-python`
*   `numpy`
*   `scipy` (Opsiyonel, daha iyi erozyon işlemi için)
*   `trimesh` (Opsiyonel, daha hassas vokselleştirme için)

## ▶️ Kullanım

```bash
python haritlama/zed3/zedtasolcum12.py
```
