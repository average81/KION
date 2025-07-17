import scipy.io
import pandas as pd
import dlib
from scipy.spatial import distance # Библиотека для вычисления евклидова расстояния между векторами признаков
from skimage import io # Библиотека для доступа к картинкам
import os
from PIL import Image, ImageDraw
import numpy as np

#Класс работы с изображениями
class FaceRecognition:
    def __init__(self, detector = 'hog'):
        if detector == 'hog':
            self.detector_type = 'hog'
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        #Составление таблицы имен и ссылок на файлы
        if not os.path.exists('../face_dataset'):
            df = pd.read_csv('data/imdb_crop.csv')
            df['image'] = df['image'].apply(lambda x: 'data/' + x)

    #Функция для определения лица на изображении
    def detect_face(self, image):
        dets = self.detector(image, 1)
        shape = None
        if len(dets) != 0 and self.detector_type == 'hog':
            for k, d in enumerate(dets):
                #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #    k, d.left(), d.top(), d.right(), d.bottom()))
                shape = self.predictor(image, d)
        return shape

    #Функция загрузки датасета
    def load_dataset(self, tomemory = False):
        if os.path.exists('../face_dataset'):
            self.data = []
            if os.path.exists('../face_dataframe/table.parquet'):
                self.datatable = pd.read_parquet('../face_dataframe/table.parquet')
                if tomemory:
                    for i in range(len(self.datatable)):
                        image = Image.open('../face_dataset/' + self.datatable['path'][i]).resize((180, 180))
                        #Если изображение черно-белое, то преобразуем в цветное
                        image = image.convert('RGB')
                        self.data.append(image)
            else:
                self.datatable = pd.DataFrame(columns=['name', 'path'])
                for path_dir in sorted(os.listdir(path='../face_dataset')):
                    path = '../face_dataset/' + path_dir + '/'
                    for path_image in sorted(os.listdir(path=path)):
                        if tomemory:
                            image = Image.open(path + path_image).resize((180, 180))
                            image = image.convert('RGB')
                            self.data.append(image)
                        new_row = {'name': path_dir, 'path': path + path_image}
                        self.datatable.loc[len(self.datatable)] = new_row
                #Сохраняем таблицу в формате parquet
                self.datatable.to_parquet('../face_dataframe/table.parquet')
                print('Создана новая таблица')


if __name__ == '__main__':
    facerec = FaceRecognition()
    facerec.load_dataset(tomemory = True)
    img1 = io.imread('https://biography-life.ru/uploads/posts/2018-09/1536266366_tom-kruz2.jpg')
    print(facerec.detect_face(img1))