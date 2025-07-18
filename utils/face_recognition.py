import scipy.io
import pandas as pd
import dlib
from scipy.spatial import distance # Библиотека для вычисления евклидова расстояния между векторами признаков
from skimage import io # Библиотека для доступа к картинкам
import os
from PIL import Image, ImageDraw
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import time
import threading
from queue import Queue


#Класс работы с изображениями
class FaceRecognition:
    def __init__(self, detector = 'hog', recognition_value = 0.5):
        if detector == 'hog':
            self.detector_type = 'hog'
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        self.recognition_value = recognition_value
        self.names = []
        self.data = pd.DataFrame(columns = ['desc'])
        self.datatable = pd.DataFrame(columns=['name', 'path'])

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
        print('Подготовка данных')

        if os.path.exists('../face_dataset'):
            if os.path.exists('../face_dataframe/table.parquet'):
                self.datatable = pd.read_parquet('../face_dataframe/table.parquet')
                self.names = self.datatable['name'].unique()
                if tomemory:
                    print('Загрузка дескрипторов лиц в ОЗУ')
                    if os.path.exists('../face_dataframe/table_desc.pcl'):
                        self.data = pd.read_pickle('../face_dataframe/table_desc.pcl')
                    else:
                        for i in tqdm.tqdm(range(len(self.datatable))):
                            image = Image.open('../face_dataset/' + self.datatable['path'][i])
                            #Если изображение черно-белое, то преобразуем в цветное
                            image = image.convert('RGB')
                            image = np.array(image)[:, :, :3]
                            descriptor = self.face_descriptor(image)
                            self.data.loc[i] = [descriptor]
                        self.data.to_pickle('../face_dataframe/table_desc.pcl')
            else:
                print('Создание таблицы для работы с данными')
                for path_dir in sorted(os.listdir(path='../face_dataset')):
                    path = '../face_dataset/' + path_dir + '/'
                    for path_image in tqdm.tqdm(sorted(os.listdir(path=path))):
                        if tomemory:
                            image = Image.open(path + path_image)
                            image = image.convert('RGB')
                            image = np.array(image)[:, :, :3]
                            descriptor = self.face_descriptor(image)
                            self.data.loc[len(self.datatable)] = [descriptor]
                        new_row = {'name': path_dir, 'path': path + path_image}
                        self.datatable.loc[len(self.datatable)] = new_row
                self.names = self.datatable['name'].unique()
                #Сохраняем таблицу в формате parquet
                self.datatable.to_parquet('../face_dataframe/table.parquet')
                self.data.to_pickle('../face_dataframe/table_desc.pcl')
                print('Создана новая таблица')

    #Функция получения рамки лица
    def shape_of_image(self,img):
        dets = self.detector(img, 1)
        shape = None
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))
            shape = self.predictor(img, d)
        return shape

    #Функция получения рамки лица для многопоточного варианта
    def shape_of_image_thread(self,image_queue, result_queue):
        while True:
            # Получаем задание из очереди
            image = image_queue.get()

            if image is None:
                break

            dets = self.detector(image, 1)
            for k, d in enumerate(dets):
                #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #    k, d.left(), d.top(), d.right(), d.bottom()))
                shape = self.predictor(image, d)
                result_queue.put((image.copy(), shape))
        image_queue.task_done()

    #Функция получения дескриптора лица
    def face_descriptor(self,img):
        shape = self.shape_of_image(img)
        if shape != None:
            return self.facerec.compute_face_descriptor(img, shape)
        return None

    #Функция сравнения 2 дескрипторов
    def dist_bool(self,face_descriptor1, face_descriptor2):
        distance = self.dist_euqlid(face_descriptor1, face_descriptor2)
        if distance < self.recognition_value:
            return True,distance
        return False,0

    #Функция сравнения 2 дескрипторов, возвращает расстояние между дескрипторами
    def dist_euqlid(self,face_descriptor1, face_descriptor2):
        if face_descriptor1 != None and face_descriptor2 != None:
            return distance.euclidean(face_descriptor1, face_descriptor2)
        return None

    #Функция сравнения 2 изображений
    def face_compare(self, img1, img2):
        return self.dist_bool(self.face_descriptor(img1), self.face_descriptor(img2))

    #Функция сравнения изображения с дескриптором
    def face_compare_img_w_desc(self,img1, descriptor):
        return self.dist_bool(self.face_descriptor(img1), descriptor)

    #Функция сравнения изображения с дескриптором
    def face_compare_w_desc(self,desc1, desc2):
        return self.dist_bool(desc1, desc2)

    #Функция поиска имени актера по фото
    def find_name(self, img):
        starttm = time.time()
        img1_desc = facerec.face_descriptor(img)
        #Предварительный поиск
        candidates = []
        for name in self.names:
            index = self.datatable[self.datatable['name']== name].iloc[0].name
            if len(self.data) != 0:
                img2_desc = self.data['desc'][index]
            else:
                path = self.datatable['path'][index]
                img2 = Image.open('../face_dataset/' + path)
                img2 = img2.convert('RGB')
                img2 = np.array(img2)[:, :, :3]
                img2_desc = self.face_descriptor(img2)
            score = self.dist_euqlid(img1_desc, img2_desc)
            print(f'"{name}",distance: {score}')
            if score < 0.7:
                candidates = candidates +[name]
                if score < self.recognition_value:
                    endtm = time.time()
                    print(f'time: {endtm-starttm},distance: {score}')
                    return name
        print(f'Кандидаты на тщательный поиск: {candidates}')
        for index,actor in self.datatable[self.datatable['name'].isin(candidates)].iterrows():
            #print(index)
            if len(self.data) != 0:
                img2_desc = self.data['desc'][index]
            else:
                path = actor['path']
                img2 = Image.open('../face_dataset/' + path)
                img2 = img2.convert('RGB')
                img2 = np.array(img2)[:, :, :3]
                img2_desc = self.face_descriptor(img2)
            result,score = self.face_compare_w_desc(img1_desc, img2_desc)
            if result:
                endtm = time.time()
                print(f'time: {endtm-starttm},distance: {score}')
                return actor['name']
        return 'Unknown'

def plot_img(img1, img2):
    height_max = max(img1.shape[1], img2.shape[1])
    width_max = max(img1.shape[0], img2.shape[0])
    fig, ax = plt.subplots(1, 2)
    w, h = figaspect(width_max/(2*height_max))
    fig.set_size_inches(w, h)
    plt.subplot(121),plt.imshow(img1)
    plt.title('Первое фото'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img2)
    plt.title('Второе фото'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':

    print(f'Использование GPU в Dlib: {dlib.DLIB_USE_CUDA}')
    facerec = FaceRecognition()
    facerec.load_dataset(tomemory = True)
    img1 = io.imread('https://biography-life.ru/uploads/posts/2018-09/1536266366_tom-kruz2.jpg')
    img1_desc = facerec.face_descriptor(img1)
    img2_id = facerec.datatable[facerec.datatable['name']=='Tom Cruise'].iloc[0].name
    print(img2_id)
    img2_desc = facerec.data['desc'][img2_id]
    print(facerec.face_compare_w_desc(img1_desc,img2_desc))

    print(facerec.find_name(img1))