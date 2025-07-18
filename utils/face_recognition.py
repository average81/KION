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


#Класс работы с изображениями
class FaceRecognition:
    def __init__(self, detector = 'hog', recognition_value = 0.5):
        if detector == 'hog':
            self.detector_type = 'hog'
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        self.recognition_value = recognition_value
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
        print('Подготовка данных')
        if os.path.exists('../face_dataset'):
            self.data = []
            if os.path.exists('../face_dataframe/table.parquet'):
                self.datatable = pd.read_parquet('../face_dataframe/table.parquet')
                if tomemory:
                    print('Загрузка изображений лиц в ОЗУ')
                    for i in tqdm.tqdm(range(len(self.datatable))):
                        image = Image.open('../face_dataset/' + self.datatable['path'][i])
                        #Если изображение черно-белое, то преобразуем в цветное
                        image = image.convert('RGB')
                        image = np.array(image)[:, :, :3]
                        descriptor = self.face_descriptor(image)
                        self.data.append(descriptor)
            else:
                print('Создание таблицы для работы с данными')
                self.datatable = pd.DataFrame(columns=['name', 'path'])
                for path_dir in sorted(os.listdir(path='../face_dataset')):
                    path = '../face_dataset/' + path_dir + '/'
                    for path_image in tqdm.tqdm(sorted(os.listdir(path=path))):
                        if tomemory:
                            image = Image.open(path + path_image)
                            image = image.convert('RGB')
                            image = np.array(image)[:, :, :3]
                            descriptor = self.face_descriptor(image)
                            self.data.append(descriptor)
                        new_row = {'name': path_dir, 'path': path + path_image}
                        self.datatable.loc[len(self.datatable)] = new_row
                #Сохраняем таблицу в формате parquet
                self.datatable.to_parquet('../face_dataframe/table.parquet')
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

    #Функция получения дескриптора лица
    def face_descriptor(self,img):
        shape = self.shape_of_image(img)
        if shape != None:
            return self.facerec.compute_face_descriptor(img, shape)
        return None

    #Функция сравнения 2 дескрипторов
    def dist_bool(self,face_descriptor1, face_descriptor2):
        if face_descriptor1 != None and face_descriptor2 != None:
            a = distance.euclidean(face_descriptor1, face_descriptor2)
            print(a)
            if a < self.recognition_value:
                print(a)
                return True
        return False

    #Функция сравнения 2 изображений
    def face_compare(self, img1, img2):
        return self.dist_bool(self.face_descriptor(img1), self.face_descriptor(img2))

    #Функция сравнения изображения с дескриптором
    def face_compare_w_desc(self,img1, descriptor):
        return self.dist_bool(self.face_descriptor(img1), descriptor)

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
    img2_id = facerec.datatable[facerec.datatable['name']=='Tom Cruise'].iloc[0].name
    print(img2_id)
    img2_desc = facerec.data[img2_id]
    print(facerec.face_compare_w_desc(img1,img2_desc))
    #plot_img(img1,img2)