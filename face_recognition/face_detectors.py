import dlib
import os
import torch
#from facenet_pytorch import InceptionResnetV1  # FaceNet для идентификации лиц

script_dir = os.path.dirname(os.path.abspath(__file__))

print(f'torch CUDA enabled:{torch.cuda.is_available()}')
print(f'cudnn version: {torch.backends.cudnn.version()}')

class HogFaceDetector:
    def __init__(self):
        self.detector_type = 'hog'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(script_dir + '/shape_predictor_68_face_landmarks.dat')

    #Функция получения рамки лица
    def shape_of_image(self,img):
        resize = 1
        if img.shape[0] < 80 or img.shape[1] < 80:
            resize = 80 / min(img.shape[0], img.shape[1])
        dets = self.detector(img, resize)
        #print("Number of faces detected: {}".format(len(dets)))
        shape = None
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))
            shape = self.predictor(img, d)
            #print(shape)
        return shape

class MmodFaceDetector:
    def __init__(self):
        self.detector_type = 'mmod'
        self.detector = dlib.cnn_face_detection_model_v1(script_dir + '/mmod_human_face_detector.dat')
        self.predictor = dlib.shape_predictor(script_dir + '/shape_predictor_68_face_landmarks.dat')

    #Функция получения рамки лица
    def shape_of_image(self,img):
        resize = 1
        if img.shape[0] < 80 or img.shape[1] < 80:
            resize = 80 // min(img.shape[0], img.shape[1])
        dets = self.detector(img, resize)
        #print("Number of faces detected: {}".format(len(dets)))
        #print(f'confidence: {[dets[i].confidence for i in range(len(dets))]}')
        shape = None
        score = 0
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()))
            if d.confidence > score:
                score = d.confidence
                shape = self.predictor(img, d.rect)
        return shape
    #Функция получения списка рамок лиц

    def shape_of_images(self,img_list):
        resize = 1
        dets_list = self.detector(img_list, 1)
        shapes = []
        shape = None
        for i in range(len(img_list)):
            shape_list = dlib.full_object_detections()
            score = 0
            for k, d in enumerate(dets_list[i]):
                #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #    k, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()))
                if d.confidence > score:
                    score = d.confidence
                    shape = self.predictor(img_list[i], d.rect)
            shape_list.append(shape)
            shapes.append(shape_list)
        return shapes
"""
class FaceNetDetector:
    def __init__(self):
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.detector_type = 'facenet'
        self.detector = InceptionResnetV1(pretrained='vggface2').eval()

    #Функция для определения лица на изображении
    def detect_face(self, image):
        if self.device == 'cuda':
            return self.detector(image.unsqueeze(0).to(self.device))
        else:
            return self.detector(image.unsqueeze(0))

    def shape_of_image(self,img):
        if self.device == 'cuda':
            return self.detector(img.unsqueeze(0).to(self.device))
        else:
            return self.detector(img.unsqueeze(0))
"""