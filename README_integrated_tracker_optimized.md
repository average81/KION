# KION - Система отслеживания объектов в видео

## Описание integrated_tracker_optimized.py

Система отслеживания объектов в видео с использованием YOLO и оптимизированного трекера. Включает в себя детекцию смен сцен и алгоритм трекинга на основе IoU.

## Архитектура

### Основные компоненты:

**`integrated_tracker_optimized.py`** - Оптимизированный трекер объектов
- Детекция смен сцен
- Детекция объектов с помощью YOLO
- Алгоритм трекинга на основе IoU
- Интерполяция позиций объектов
- Фильтрация дубликатов

## Быстрый старт

### Настройка входного файла:

1. **Скачайте тестовое видео** с https://cloud.mail.ru/public/LpXj/G1euCFYrn
2. **Поместите видео файл** в корневую папку проекта
3. **Измените имя файла** в коде или переименуйте файл в `mister-i-missis-smit-2005_1.mkv`
4. **Поддерживаемые форматы**: `.mkv`, `.mp4`, `.avi`, `.mov`
5. **Перед запуском скачайте необходимые модели YOLO и поместите в корень папки** с https://cloud.mail.ru/public/ubai/DUgAKHn9X


```python
# В файле integrated_tracker_optimized.py, строка 691:
def main():
    processor = OptimizedVideoProcessor()
    processor.process_video("mister-i-missis-smit-2005_1.mkv", "output_video_optimized.mp4")
```

### Запуск обработки:

```bash
# Обработка видео с трекером
python integrated_tracker_optimized.py
```

## Возможности

- Быстрая обработка видео
- Детекция смен сцен для сброса трекера
- Детекция людей с помощью YOLO
- Отслеживание объектов с уникальными ID
- Сохранение аудио в выходном видео
- Минимум зависимостей

## Классы и параметры

### `OptimizedVideoProcessor`
Главный класс для обработки видео.

**Параметры инициализации:**
```python
processor = OptimizedVideoProcessor(model_path="yolov8n.pt")
```

**Основные методы:**
- `process_video(input_path, output_path)` - обработка видео
- `_filter_duplicates(detections)` - фильтрация дубликатов
- `_draw_results(frame, tracked_boxes, frame_num, fps)` - отрисовка результатов

### `OptimizedBoundingBoxTracker`
Трекер объектов с улучшенной производительностью.

**Параметры инициализации:**
```python
tracker = OptimizedBoundingBoxTracker(
    max_history=30,                    # История позиций объекта
    interpolation_frames=20,           # Кадры для интерполяции
    iou_threshold=0.3,                 # Порог IoU для сопоставления
    min_detections_for_tracking=6,     # Минимум детекций для активации
    min_confidence=0.7                 # Минимальная уверенность
)
```

**Примеры настроек трекера:**

**Для быстрого движения объектов:**
```python
tracker = OptimizedBoundingBoxTracker(
    max_history=20,                    # Меньшая история для быстрых движений
    interpolation_frames=10,           # Меньше кадров интерполяции
    iou_threshold=0.2,                 # Более низкий порог IoU
    min_detections_for_tracking=4,     # Быстрая активация трекинга
    min_confidence=0.6                 # Более низкая уверенность
)
```

**Для медленного движения объектов:**
```python
tracker = OptimizedBoundingBoxTracker(
    max_history=50,                    # Большая история для плавности
    interpolation_frames=30,           # Больше кадров интерполяции
    iou_threshold=0.4,                 # Более высокий порог IoU
    min_detections_for_tracking=8,     # Более стабильная активация
    min_confidence=0.8                 # Высокая уверенность
)
```

**Для переполненных сцен:**
```python
tracker = OptimizedBoundingBoxTracker(
    max_history=25,                    # Средняя история
    interpolation_frames=15,           # Средняя интерполяция
    iou_threshold=0.25,                # Низкий порог для перекрытий
    min_detections_for_tracking=10,    # Много детекций для стабильности
    min_confidence=0.75                # Средняя уверенность
)
```

**Основные методы:**
- `update(detections, frame_num)` - обновление трекера
- `_match_detections(detections)` - сопоставление детекций
- `_remove_expired_objects()` - удаление устаревших объектов
- `reset()` - сброс трекера при смене сцены

### `TrackedObject`
Класс для отслеживания отдельного объекта.

**Параметры инициализации:**
```python
obj = TrackedObject(
    object_id,              # Уникальный ID объекта
    initial_detection,      # Первая детекция
    creation_frame,         # Кадр создания
    max_history,            # Максимальная история
    interpolation_frames    # Кадры интерполяции
)
```

**Основные методы:**
- `update(detection, frame_num)` - обновление объекта
- `get_smoothed_box()` - получение сглаженной рамки
- `get_interpolated_box(current_frame)` - интерполированная рамка
- `check_interpolation(current_frame)` - проверка интерполяции

### `OptimizedSceneDetector`
Детектор смен сцен.

**Принцип работы:**
1. **Вычисление разности кадров** - сравнивает текущий кадр с предыдущим
2. **Нормализация разности** - приводит к диапазону 0-100%
3. **Проверка порога** - если разность > threshold, считает сменой сцены
4. **Минимальная длина сцены** - игнорирует слишком короткие сцены
5. **Последовательные детекции** - требует несколько кадров подряд для подтверждения

**Параметры инициализации:**
```python
detector = OptimizedSceneDetector(
    threshold=10.0,         # Порог детекции смены сцены (0-100%)
    min_scene_length=0.7,   # Минимальная длина сцены (секунды)
    required_consecutive=1  # Требуемые последовательные детекции
)
```

**Алгоритм детекции:**
```python
# 1. Конвертация в оттенки серого
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 2. Вычисление разности с предыдущим кадром
diff = cv2.absdiff(gray, self.prev_frame)

# 3. Нормализация разности
mean_diff = np.mean(diff)
normalized_diff = (mean_diff / 255.0) * 100

# 4. Проверка условий
if (normalized_diff > self.threshold and 
    frame_num - self.last_scene_change >= min_frames):
    # Смена сцены детектирована
```

**Примеры настроек детектора сцен:**

**Для фильмов с четкими сменами сцен:**
```python
detector = OptimizedSceneDetector(
    threshold=8.0,          # Низкий порог для четких смен
    min_scene_length=1.0,   # Минимум 1 секунда сцены
    required_consecutive=1  # Мгновенная детекция
)
```

**Для видео с плавными переходами:**
```python
detector = OptimizedSceneDetector(
    threshold=15.0,         # Высокий порог для плавных переходов
    min_scene_length=2.0,   # Минимум 2 секунды сцены
    required_consecutive=3  # Требует 3 кадра подряд
)
```

**Для видео с частыми изменениями освещения:**
```python
detector = OptimizedSceneDetector(
    threshold=20.0,         # Очень высокий порог
    min_scene_length=0.5,   # Короткие сцены разрешены
    required_consecutive=5  # Много подтверждений
)
```

**Примеры работы детектора сцен:**

**Сцена 1: Плавный переход (не детектируется)**
```
Кадр 1: normalized_diff = 2.1%  (ниже порога 10%)
Кадр 2: normalized_diff = 1.8%  (ниже порога 10%)
Кадр 3: normalized_diff = 2.3%  (ниже порога 10%)
Результат: Смена сцены НЕ детектирована
```

**Сцена 2: Резкая смена (детектируется)**
```
Кадр 1: normalized_diff = 2.1%  (ниже порога 10%)
Кадр 2: normalized_diff = 15.3% (выше порога 10%) → consecutive_detections = 1
Результат: Смена сцены детектирована (required_consecutive = 1)
```

**Сцена 3: Строгая детекция (требует подтверждения)**
```
Кадр 1: normalized_diff = 12.1% (выше порога 10%) → consecutive_detections = 1
Кадр 2: normalized_diff = 11.8% (выше порога 10%) → consecutive_detections = 2  
Кадр 3: normalized_diff = 13.2% (выше порога 10%) → consecutive_detections = 3
Результат: Смена сцены детектирована (required_consecutive = 3)
```

**Основные методы:**
- `detect_scene_change(frame, frame_num, fps)` - детекция смены сцены
- `get_current_scene(frame_num)` - получение номера текущей сцены
- `save_analysis(output_file, fps)` - сохранение анализа

## Использование

### Логика работы алгоритма:

1. **Детекция смен сцен** - определяет границы сцен для сброса трекера
2. **Детекция объектов** - YOLO находит людей в каждом кадре
3. **Трекинг объектов** - сопоставляет детекции с отслеживаемыми объектами
4. **Интерполяция** - заполняет пропуски в трекинге
5. **Фильтрация** - убирает дублирующиеся детекции

### Обработка видео:

```python
from integrated_tracker_optimized import OptimizedVideoProcessor

# Создание процессора
processor = OptimizedVideoProcessor("yolov8n.pt")

# Обработка видео
processor.process_video("input.mkv", "output.mp4")
```

### Настройка трекера:

```python
from integrated_tracker_optimized import OptimizedBoundingBoxTracker

# Настройка трекера
tracker = OptimizedBoundingBoxTracker(
    max_history=30,                    # История позиций
    interpolation_frames=20,           # Кадры интерполяции
    iou_threshold=0.3,                 # Порог IoU
    min_detections_for_tracking=6,     # Минимум детекций
    min_confidence=0.7                 # Минимальная уверенность
)
```

### Настройка детектора сцен:

```python
from integrated_tracker_optimized import OptimizedSceneDetector

# Настройка детектора сцен
detector = OptimizedSceneDetector(
    threshold=10.0,         # Порог детекции (0-100%)
    min_scene_length=0.7,   # Минимальная длина сцены (секунды)
    required_consecutive=1  # Требуемые детекции подряд
)
```

## Выходные данные

Система создает:
- Обработанное видео с рамками и ID объектов
- CSV файл с анализом сцен (`scene_analysis_optimized.csv`)
- Видео с аудио - сохраняет оригинальную звуковую дорожку

## Установка

```bash
pip install -r requirements.txt
```

## Структура проекта

```
KION/
├── integrated_tracker_optimized.py    # Основной трекер
├── requirements.txt                   # Зависимости
├── mister-i-missis-smit-2005_1.mkv   # Входное видео (добавить самостоятельно)
└── output/                           # Результаты
```

## Отладка

### Логирование:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Настройка детектора сцен:
- **Увеличьте `threshold`** (например, до 15.0) для менее чувствительной детекции
- **Уменьшите `threshold`** (например, до 5.0) для более чувствительной детекции
- **Увеличьте `min_scene_length`** для игнорирования коротких сцен

### Настройка трекера:
- **Увеличьте `iou_threshold`** для более строгого сопоставления объектов
- **Уменьшите `min_detections_for_tracking`** для быстрой активации трекинга
- **Увеличьте `interpolation_frames`** для более плавной интерполяции