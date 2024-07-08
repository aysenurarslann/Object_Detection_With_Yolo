# YOLOv8 Object Tracking with SORT

Bu proje, YOLOv8 kullanarak video üzerindeki nesneleri tespit eder ve SORT algoritması ile takip eder. Aynı zamanda, belirli bir çizgiyi geçen nesneleri sayar ve sayacı video üzerinde görüntüler.

## Gereksinimler

- Python 3.8 veya daha üstü
- OpenCV
- Ultralytics YOLO
- NumPy
- SORT (Simple Online and Realtime Tracking) https://github.com/abewley/sort

## Kurulum

Öncelikle, gerekli Python kütüphanelerini yükleyin:

```bash
pip install opencv-python-headless ultralytics numpy
