from face_recognition import FaceRecognition
from tracker import *
import logging
import argparse
import time
import os

if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="парсер параметров")
    parser.add_argument('--video', type=str, required=True, help='входной файл видео')
    parser.add_argument('--output_dir', type=str, required=False, help='директория для сохранения результата работы')
    parser.add_argument('-update_base', action='store_true', required=False, help='принудительное обновление датасета лиц')
    parser.add_argument('--vtracker', type=str, required=False, help='Тип трекера (yolo, custom)')
    # Получаем аргумент из командной строки
    args = parser.parse_args()
    #Подготовка модулей
    logging.info('Start')
    #face_recognition = FaceRecognition()
    #Создание выходной папки
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    if args.vtracker:
        if args.vtracker == 'yolo':
            logging.info('Using YOLO tracker')
            processor = ComplexVideoProcessor(force_update = args.update_base)
        elif args.vtracker == 'custom':
            logging.info('Using Custom tracker')
            tracker = processor = OptimizedVideoProcessor(force_update = args.update_base)
        else:
            logging.info('Using YOLO tracker')
            processor = ComplexVideoProcessor(force_update = args.update_base)
    else:
        logging.info('Using YOLO tracker')
        processor = ComplexVideoProcessor(force_update = args.update_base)
    processor.process_video(args.video, args.output_dir)
    end_time = time.time()
    logging.info(f'Total Time: {end_time - start_time}')
