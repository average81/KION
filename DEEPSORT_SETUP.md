# Настройка DeepSORT

## 🚫 Исключенные файлы

Следующие файлы и папки **НЕ загружаются в Git**:

- `deep_sort_pytorch/` - вся папка DeepSORT
- `*.yaml`, `*.yml` - конфигурационные файлы
- `*.pt`, `*.pth`, `*.weights` - веса моделей
- `*.log`, `*.tmp` - временные файлы
- `output/`, `results/` - папки с результатами

## 📋 Что нужно сделать вручную

### 1. Скачать веса моделей

```bash
# Создайте папки для весов
mkdir -p deep_sort_pytorch/deep_sort/deep/checkpoint
mkdir -p deep_sort_pytorch/detector/YOLOv3/weight
mkdir -p deep_sort_pytorch/detector/YOLOv5
mkdir -p deep_sort_pytorch/detector/Mask_RCNN/save_weights

# Скачайте веса ReID модели
cd deep_sort_pytorch/deep_sort/deep/checkpoint
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
# Или для оригинальной модели:
# wget https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8n_riXi6

# Скачайте веса YOLOv3
cd ../../../detector/YOLOv3/weight
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights

# Скачайте веса YOLOv5
cd ../../YOLOv5
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt

# Скачайте веса Mask RCNN
cd ../Mask_RCNN/save_weights
wget https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
```

### 2. Создать конфигурационные файлы

```bash
# Скопируйте пример конфигурации
cp deep_sort_pytorch/configs/deep_sort_example.yaml deep_sort_pytorch/configs/deep_sort.yaml

# Отредактируйте пути в файле
nano deep_sort_pytorch/configs/deep_sort.yaml
```

### 3. Установить зависимости

```bash
# Основные зависимости
pip install torch torchvision
pip install opencv-python numpy scipy sklearn
pip install pillow matplotlib tqdm

# Для DeepSORT
cd deep_sort_pytorch
pip install -r requirements.txt

# Для YOLOv5
pip install ultralytics

# Для Mask RCNN
pip install pycocotools
```

## 🔧 Настройка конфигурации

### Пример `deep_sort.yaml`:

```yaml
DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/resnet18-5c106cde.pth"
  MAX_DIST: 0.5
  MIN_CONFIDENCE: 0.5
  NMS_MAX_OVERLAP: 0.8
  MAX_IOU_DISTANCE: 0.8
  MAX_AGE: 500
  N_INIT: 1
  NN_BUDGET: 100

USE_FASTREID: False

YOLOV5:
  WEIGHT: "yolov5s.pt"
  DATA: "coco.yaml"
  IMGSZ: 640
  SCORE_THRESH: 0.5
  NMS_THRESH: 0.4
  MAX_DET: 15
```

## 🚀 Запуск

### Базовый запуск:
```bash
python deep_sort_pytorch/deepsort.py video.mp4 --config_detection deep_sort_pytorch/configs/yolov5s.yaml
```

### С веб-камерой:
```bash
python deep_sort_pytorch/deepsort.py /dev/video0 --camera 0
```

### С отображением:
```bash
python deep_sort_pytorch/deepsort.py video.mp4 --display
```

## 📁 Структура файлов

```
deep_sort_pytorch/
├── configs/
│   ├── deep_sort_example.yaml  # ✅ В Git (пример)
│   └── deep_sort.yaml          # ❌ НЕ в Git (ваша конфигурация)
├── deep_sort/
│   └── deep/
│       └── checkpoint/
│           └── resnet18-5c106cde.pth  # ❌ НЕ в Git (веса)
├── detector/
│   ├── YOLOv3/
│   │   └── weight/
│   │       └── yolov3.weights  # ❌ НЕ в Git (веса)
│   └── YOLOv5/
│       └── yolov5s.pt          # ❌ НЕ в Git (веса)
└── deepsort.py                 # ✅ В Git (код)
```

## ⚠️ Важные замечания

1. **Веса моделей** не загружаются в Git из-за большого размера
2. **Конфигурационные файлы** могут содержать пути к локальным файлам
3. **Временные файлы** и логи исключены для чистоты репозитория
4. **Результаты** сохраняются в папку `output/` (исключена из Git)

## 🔍 Проверка настройки

```bash
# Проверьте, что все файлы на месте
ls -la deep_sort_pytorch/deep_sort/deep/checkpoint/
ls -la deep_sort_pytorch/detector/YOLOv3/weight/
ls -la deep_sort_pytorch/detector/YOLOv5/

# Проверьте конфигурацию
python -c "import yaml; yaml.safe_load(open('deep_sort_pytorch/configs/deep_sort.yaml'))"
``` 