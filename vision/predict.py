import cv2
import numpy as np
from PIL import Image

import torchvision.transforms as transforms

from utils import image_resize
from vision.vgg import *

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_model(path):
    net = VGG('VGG19')
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return net

def predict_vgg(model, cvimg, haarcascade_path, debug=False):
    class_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    cvimg = image_resize(cvimg, width=100)
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    faces = face_cascade.detectMultiScale(cvimg)

    # print(f"Time to compute faces: {time.time() - start}")
    score = torch.zeros([7])
    for (x, y, w, h) in faces:
        resize_frame = cv2.resize(cvimg[y:y + h, x:x + w], (48, 48))
        img = Image.fromarray(resize_frame)
        with torch.no_grad():
            inputs = transform_test(img)

            ncrops, c, h, w = np.shape(inputs)

            inputs = inputs.view(-1, c, h, w)
            inputs = Variable(inputs)
            outputs = model(inputs)

            outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

            score = F.softmax(outputs_avg)
            _, predicted = torch.max(outputs_avg.data, 0)
            if debug:
                print(score, _, predicted, class_names[int(predicted.cpu().numpy())])
            return score #class_names[int(predicted.cpu().numpy())]
    return score