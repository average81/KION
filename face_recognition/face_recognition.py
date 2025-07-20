import logging
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
from face_detectors import HogFaceDetector, MmodFaceDetector

#Список детекторов
face_detectors = {
    'hog': HogFaceDetector,
    'mmod': MmodFaceDetector,
}

script_dir = os.path.dirname(os.path.abspath(__file__))
#Класс работы с изображениями
class FaceRecognition:
    def __init__(self, detector = 'hog', recognition_value = 0.5):
        if detector not in face_detectors:
            logging.error('Детектор не найден')
            return
        for det in face_detectors:
            if detector == det:
                self.detector = face_detectors[det]()
        logging.info('Подключение детектора ' + detector + ' к классу FaceRecognition')
        self.facerec = dlib.face_recognition_model_v1(script_dir + '/dlib_face_recognition_resnet_model_v1.dat')
        self.recognition_value = recognition_value
        self.names = []
        self.data = pd.DataFrame(columns = ['desc'])
        self.datatable = pd.DataFrame(columns=['name', 'path'])

    #Функция загрузки датасета
    def load_dataset(self, tomemory = False):
        logging.info('Подготовка данных')

        if os.path.exists(script_dir + '/../face_dataset'):
            if os.path.exists(script_dir + '/../face_dataframe/table.parquet'):
                self.datatable = pd.read_parquet(script_dir + '/../face_dataframe/table.parquet')
                self.names = self.datatable['name'].unique()
                if tomemory:
                    logging.info('Загрузка дескрипторов лиц в ОЗУ')
                    if os.path.exists(script_dir + '/../face_dataframe/table_desc.pcl'):
                        self.data = pd.read_pickle(script_dir + '/../face_dataframe/table_desc.pcl')
                    else:
                        for i in tqdm.tqdm(range(len(self.datatable))):
                            image = Image.open(script_dir + '/../face_dataset/' + self.datatable['path'][i])
                            #Если изображение черно-белое, то преобразуем в цветное
                            image = image.convert('RGB')
                            image = np.array(image)[:, :, :3]
                            descriptor = self.face_descriptor(image)
                            self.data.loc[i] = [descriptor]
                        self.data.to_pickle(script_dir + '/../face_dataframe/table_desc.pcl')

            else:
                logging.info('Создание таблицы для работы с данными')
                for path_dir in sorted(os.listdir(path=script_dir + '/../face_dataset')):
                    path = script_dir + '/../face_dataset/' + path_dir + '/'
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
                self.datatable.to_parquet(script_dir + '/../face_dataframe/table.parquet')
                self.data.to_pickle(script_dir + '/../face_dataframe/table_desc.pcl')
                logging.info('Создана новая таблица')

    #Функция получения дескриптора лица
    def face_descriptor(self,img):
        shape = self.detector.shape_of_image(img)
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
        img1_desc = self.face_descriptor(img)
        #Предварительный поиск
        candidates = []
        for name in self.names:
            index = self.datatable[self.datatable['name']== name].iloc[0].name
            if len(self.data) != 0:
                img2_desc = self.data['desc'][index]
            else:
                path = self.datatable['path'][index]
                img2 = Image.open(script_dir + '/../face_dataset/' + path)
                img2 = img2.convert('RGB')
                img2 = np.array(img2)[:, :, :3]
                img2_desc = self.face_descriptor(img2)
            score = self.dist_euqlid(img1_desc, img2_desc)
            logging.info(f'"{name}",distance: {score}')
            if score < 0.7:
                candidates = candidates +[name]
                if score < self.recognition_value:
                    endtm = time.time()
                    print(f'time: {endtm-starttm},distance: {score}')
                    return name
        logging.info(f'Кандидаты на тщательный поиск: {candidates}')
        for index,actor in self.datatable[self.datatable['name'].isin(candidates)].iterrows():
            #print(index)
            if len(self.data) != 0:
                img2_desc = self.data['desc'][index]
            else:
                path = actor['path']
                img2 = Image.open(script_dir + '/../face_dataset/' + path)
                img2 = img2.convert('RGB')
                img2 = np.array(img2)[:, :, :3]
                img2_desc = self.face_descriptor(img2)
            result,score = self.face_compare_w_desc(img1_desc, img2_desc)
            if result:
                endtm = time.time()
                logging.info(f'time: {endtm-starttm},distance: {score}')
                return actor['name']
        return 'Unknown'

    #Функция поиска имен актеров по фото
    def find_names(self, images, path = None):
        #images - словарь с фото и ID актеров
        #path - путь к папке с фото
        starttm = time.time()
        names = {}  # Словарь с именами и дескрипторами лиц
        if len(images) == 0:
            images = {}
            if path != None:
                for path_dir in sorted(os.listdir(path=path)):
                    path = path + path_dir + '/'
                    for path_image in sorted(os.listdir(path=path)):
                        image = Image.open(path + path_image)
                        image = image.convert('RGB')
                        image = np.array(image)[:, :, :3]
                        images[path_dir] = image
            else:
                df = pd.DataFrame(columns=['id','name'])
                logging.info('Wrong parameters for the face recognition')
                return df
        for id in images:
            for img in images[id]:
                name = self.find_name(img)
                names[id] = name
        endtm = time.time()
        logging.info(f' Faces recognition time: {endtm-starttm}')
        logging.info(names)
        df = pd.DataFrame(columns=['id','name'])
        for id,name in names.items():
            df.loc[len(df)] = [id,name]
        if not os.path.exists(script_dir + '/../tmp'):
            os.mkdir(script_dir + '/../tmp')
        df.to_csv(script_dir + '/../tmp/names.csv', index=True)
        return df


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
    logging.basicConfig(level=logging.INFO)
    print(f'Использование GPU в Dlib: {dlib.DLIB_USE_CUDA}')
    facefnd = FaceRecognition(detector = 'mmod')
    facefnd.load_dataset(tomemory = True)
    img1 = io.imread('https://biography-life.ru/uploads/posts/2018-09/1536266366_tom-kruz2.jpg')
    logging.info('Фото загружено')
    print('----------')
    img1_desc = facefnd.face_descriptor(img1)
    logging.info('Дескриптор фото посчитан')
    img2_id = facefnd.datatable[facefnd.datatable['name']=='Tom Cruise'].iloc[0].name
    print(img2_id)
    img2_desc = facefnd.data['desc'][img2_id]
    images = {'0':[img1]}
    print(facefnd.find_names(images))