import logging
import pandas as pd
import dlib
from scipy.spatial import distance # Библиотека для вычисления евклидова расстояния между векторами признаков
from skimage import io # Библиотека для доступа к картинкам
import os
import tqdm
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import time
from .face_detectors import HogFaceDetector, MmodFaceDetector

#Список детекторов
face_detectors = {
    'hog': HogFaceDetector,
    'mmod': MmodFaceDetector,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.local_dataset = pd.DataFrame(columns=['name', 'desc'])

    #Функция для добавления дескриптора лица в локальный датасет
    def add_face_desc(self, desc, name):
        self.local_dataset.loc[len(self.local_dataset)] = [name, desc]

    #Функция загрузки датасета
    def load_dataset(self, tomemory = False, force_update = False):
        logging.info('Подготовка данных')

        if os.path.exists(script_dir + '/../face_dataset'):
            if os.path.exists(script_dir + '/../face_dataframe/table.parquet') and not force_update:
                self.datatable = pd.read_parquet(script_dir + '/../face_dataframe/table.parquet')
                self.names = self.datatable['name'].unique()
                if tomemory:
                    logging.info('Загрузка дескрипторов лиц в ОЗУ')
                    if os.path.exists(script_dir + '/../face_dataframe/table_desc.pcl'):
                        self.data = pd.read_pickle(script_dir + '/../face_dataframe/table_desc.pcl')
                    else:
                        for i in tqdm.tqdm(range(len(self.datatable))):
                            image = dlib.load_rgb_image(script_dir + '/../face_dataset/' + self.datatable['path'][i])
                            descriptor = self.face_descriptor(image)
                            self.data.loc[i] = [descriptor]
                        self.data.to_pickle(script_dir + '/../face_dataframe/table_desc.pcl')

            else:
                logging.info('Создание таблицы для работы с данными')
                for path_dir in sorted(os.listdir(path=script_dir + '/../face_dataset')):
                    path = script_dir + '/../face_dataset/' + path_dir + '/'
                    logger.info(f'Обработка папки: {path_dir}')
                    for path_image in tqdm.tqdm(sorted(os.listdir(path=path))):
                        if tomemory:
                            image = dlib.load_rgb_image(path + path_image)
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
            #Вывод рамки лица
            #print(f'shape: {shape.parts()}')
            return self.facerec.compute_face_descriptor(img, shape)
        return None

    #Функция сравнения 2 дескрипторов
    def dist_bool(self,face_descriptor1, face_descriptor2):
        distance = self.dist_euqlid(face_descriptor1, face_descriptor2)
        if distance == None:
            return False,0
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
                img2 = dlib.load_rgb_image(script_dir + '/../face_dataset/' + path)
                img2_desc = self.face_descriptor(img2)
            score = self.dist_euqlid(img1_desc, img2_desc)
            logging.debug(f'"{name}",distance: {score}')
            if score == None:
                continue
            if score < 0.7:
                candidates = candidates +[name]
                if score < self.recognition_value:
                    endtm = time.time()
                    #print(f'time: {endtm-starttm},distance: {score}')
                    return name
        logging.debug(f'Кандидаты на тщательный поиск: {candidates}')
        for index,actor in self.datatable[self.datatable['name'].isin(candidates)].iterrows():
            #print(index)
            if len(self.data) != 0:
                img2_desc = self.data['desc'][index]
            else:
                path = actor['path']
                img2 = dlib.load_rgb_image(script_dir + '/../face_dataset/' + path)
                img2_desc = self.face_descriptor(img2)
            result,score = self.face_compare_w_desc(img1_desc, img2_desc)
            if result:
                endtm = time.time()
                logging.debug(f'time: {endtm-starttm},distance: {score}')
                return actor['name']
        return 'Unknown'

    #Функция поиска имени актера по дескриптору лица
    def find_name_desc(self, desc):
        starttm = time.time()
        img1_desc = desc
        #Предварительный поиск
        #Сначала ищем в локальном датасете дескрипторов лиц
        candidates = []
        if len(self.local_dataset) != 0:
            for name in self.local_dataset['name'].unique():
                #index = self.local_dataset[self.local_dataset['name']== name].iloc[0].name
                img2_desc = self.local_dataset[self.local_dataset['name']== name].iloc[0].desc
                result, score = self.face_compare_w_desc(img1_desc, img2_desc)
                if result:
                    endtm = time.time()
                    logging.debug(f'time: {endtm-starttm},distance: {score}')
                    return name
                elif score < 0.5:
                    candidates = candidates +[name]
            for index,actor in self.local_dataset[self.local_dataset['name'].isin(candidates)].iterrows():
                img2_desc = actor['desc']
                result,score = self.face_compare_w_desc(img1_desc, img2_desc)
                if result:
                    endtm = time.time()
                    logging.debug(f'time: {endtm-starttm},distance: {score}')
                    return actor['name']
        #Если не нашли в локальном датасете, то ищем в глобальном датасете
        candidates = []
        for name in self.names:
            index = self.datatable[self.datatable['name']== name].iloc[0].name
            if len(self.data) != 0:
                img2_desc = self.data['desc'][index]
            else:
                path = self.datatable['path'][index]
                img2 = dlib.load_rgb_image(script_dir + '/../face_dataset/' + path)
                img2_desc = self.face_descriptor(img2)
            score = self.dist_euqlid(img1_desc, img2_desc)
            logging.debug(f'"{name}",distance: {score}')
            if score == None:
                continue
            if score < 0.7:
                candidates = candidates +[name]
                if score < self.recognition_value:
                    endtm = time.time()
                    #print(f'time: {endtm-starttm},distance: {score}')
                    return name
        logging.debug(f'Кандидаты на тщательный поиск: {candidates}')
        for index,actor in self.datatable[self.datatable['name'].isin(candidates)].iterrows():
            #print(index)
            if len(self.data) != 0:
                img2_desc = self.data['desc'][index]
            else:
                path = actor['path']
                img2 = dlib.load_rgb_image(script_dir + '/../face_dataset/' + path)
                img2_desc = self.face_descriptor(img2)
            result,score = self.face_compare_w_desc(img1_desc, img2_desc)
            if result:
                endtm = time.time()
                logging.debug(f'time: {endtm-starttm},distance: {score}')
                return actor['name']
        return 'Unknown'

    #Функция поиска имени актера по дескриптору лица в локальной базе
    def find_name_local_desc(self, desc):
        starttm = time.time()
        img1_desc = desc
        for actor, desc in self.local_dataset.iterrows():
            result,score = self.face_compare_w_desc(img1_desc, desc)
            if result:
                endtm = time.time()
                logging.debug(f'time: {endtm-starttm},distance: {score}')
                return actor['name']
        return 'Unknown'

    #Функция поиска имен актеров по фото
    def find_names(self, images, path = None):
        #images - словарь с фото и ID актеров
        #path - путь к папке с фото
        starttm = time.time()
        names = {}  # Словарь с именами и дескрипторами лиц
        #img_list = list(images.values())
        shapes_list = self.detector.shape_of_images(images)
        desc_list = self.facerec.compute_face_descriptor(images, shapes_list)
        names = []
        for id in range(len(images)):
            #print(desc_list[id].pop())
            name = self.find_name_desc(desc_list[id].pop(0))
            names.append(name)
        endtm = time.time()
        logging.info(f' Faces recognition time: {endtm-starttm}')
        logging.info(names)
        return names


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
    #img1 = io.imread('https://biography-life.ru/uploads/posts/2018-09/1536266366_tom-kruz2.jpg')
    img1 = dlib.load_rgb_image('tst1.png')
    logging.info('Фото загружено')
    print('----------')
    img1_desc = facefnd.face_descriptor(img1)
    logging.info('Дескриптор фото посчитан')
    img2_id = facefnd.datatable[facefnd.datatable['name']=='Tom Cruise'].iloc[0].name
    print(img2_id)
    img2_desc = facefnd.data['desc'][img2_id]
    start_time = time.time()
    images = [img1,img1,img1,img1,img1,img1,img1,img1,img1,img1]
    for i in range(10):
        facefnd.find_names(images)

    end_time = time.time()
    print(facefnd.find_names(images))
    print(end_time-start_time)
    print(facefnd.find_name(img1))