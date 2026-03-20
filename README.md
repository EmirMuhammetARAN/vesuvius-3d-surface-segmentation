# Vesuvius 3D Surface Segmentation

Bu proje, Vesuvius Challenge için 3D U-Net tabanlı yüzey segmentasyon modeli içerir. TensorFlow ile hızlı veri üretimi, özel loss ve IoU metriği, checkpoint kaydı ve eğitim/validasyon görselleştirmesi sunar.

## Dosyalar
- model.ipynb: Model eğitimi ve değerlendirme notebook'u
- checkpoints/: En iyi model ağırlıkları (model_3d_v4_best.h5 vb.)
- train_images/: Eğitim verisi (bulut linki ile paylaşılmalı)
- train_labels/: Etiket verisi (bulut linki ile paylaşılmalı)

## Kullanım
1. Notebook'u açıp çalıştırın.
2. Model eğitimi ve checkpoint kaydı yapılır.
3. Eğitim/validasyon sonuçları görselleştirilir.

## Gereksinimler
- tensorflow
- numpy
- tifffile
- matplotlib

## Veri ve Model
- Büyük veri ve model dosyalarını bulut depolama ile paylaşın, README'de link verin.

## Lisans
MIT