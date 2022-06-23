
from utils import getVideoFilesFromFolder
import time
import uuid

import sklearn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append('./pyAudioAnalysis')
from pyAudioAnalysis import audioTrainTest as aT
import cv2
import torch
import os
import numpy as np
from vision.predict import predict_vgg

class CombinedModel:
    def __init__(self, image_model, audio_model_path, audio_model_type, labels):
        self.image_model = image_model
        self.audio_model = audio_model_path
        self.audio_model_type = audio_model_type
        self.meta_clf = sklearn.ensemble.RandomForestClassifier()
        self.labels = labels
        self.no_audio = False
        self.no_visual = False

    def calculate_base_posteriors(self, no_audio, no_visual, dir_paths, labels):
        X = []
        y = []
        for dir_path, label in zip(dir_paths, labels):
            files = getVideoFilesFromFolder(dir_path)
            print("-------------------------------")
            print(f"Label: {label}")

            for i, f in enumerate(files):
                start_t = time.time()
                if not no_audio:
                    audio_features = self.extract_audio_prediction(f)
                    id_ = np.argmax(audio_features, axis=0)
                    print(f"Audio: {self.labels[id_]}")
                else:
                    audio_features = np.array([])

                if not no_visual:
                    video_features = self.extract_video_prediction(f)
                    id_ = np.argmax(video_features, axis=0)
                    print(f"Video: {self.labels[id_]}")
                else:
                    video_features = np.array([])

                features = np.concatenate((audio_features, video_features))
                X.append(features)
                y.append(label)
                print(f'Analyzed file {i} of {len(files)} in {time.time() - start_t} seconds')
        return np.array(X), np.array(y)

    def extract_video_prediction(self, video_file_path, debug=False):
        probs_array = []

        process_fn = lambda frame: probs_array.append(
            predict_vgg(self.image_model, frame,
                        haarcascade_path='./models/haarcascade_frontalface_default.xml', debug=debug))

        # Open video and process each frame
        cap = cv2.VideoCapture(video_file_path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            process_fn(frame)
            count = count + 1

        preds = torch.stack(probs_array)
        mean_preds = torch.mean(preds, dim=0).squeeze(0).numpy()
        return mean_preds/np.sum(mean_preds) #4, 1, 3 | 4,5,6,1,3,0,2

    def extract_audio_prediction(self, video_file_path):
        # Save converted file
        samplingRate = int(44100)
        channels = int(2)
        tmp_path = './tmp/'
        rand_name = str(uuid.uuid4()) + ".wav"
        ffmpegString = 'ffmpeg -y -i ' + '\"' + video_file_path + '\"' + ' -ar ' + str(samplingRate) + ' -ac ' + str(
            channels) + ' ' + tmp_path + rand_name
        os.system(ffmpegString)
        res = aT.file_classification(tmp_path + rand_name, self.audio_model, self.audio_model_type)

        # Align model labels with self labels
        ids = [res[2].index(lab) for lab in self.labels[0:5]]
        return np.array(res[1])[ids]

    def train(self, dir_paths, labels, test_dir_paths=None, test_labels=None, base_model_posteriors=None,
              save_base_model_posteriors=None, no_audio=False,
              no_visual=False):
        self.no_audio = no_audio
        self.no_visual = no_visual
        if base_model_posteriors is None:
            # Calculate base models posteriors
            X, y = self.calculate_base_posteriors(no_audio, no_visual, dir_paths, labels)
            if save_base_model_posteriors is not None:
                np.save(save_base_model_posteriors + '_posteriors.npy', X)
                np.save(save_base_model_posteriors + '_labels.npy', y)

        else:
            # Load already calculated posteriors to save time
            X = np.load(base_model_posteriors)
            y = np.load(labels)

        if test_dir_paths is not None:
            X_test, y_test = self.calculate_base_posteriors(no_audio, no_visual, test_dir_paths, test_labels)
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        print(f"Training meta classifier with {len(X_train)} samples")
        self.meta_clf.fit(X_train, y_train)

        print(f"Evaluation on test set: {len(X_test)} samples")
        y_pred = self.meta_clf.predict(X_test)

        print("------- Combined Model ------------")
        print("Confusion matrix of combined model")
        print(confusion_matrix(y_test, y_pred, labels=self.labels))

        print("f1 score")
        print(f1_score(y_test, y_pred, labels=self.labels, average='weighted'))
        print("-----------------------------------")

        if not no_audio:
            y_pred_aud_ids = np.argmax(X_test[:, 0:5], axis=1)
            y_pred_aud = [self.labels[i] for i in y_pred_aud_ids]

            print("------- Audio Model ------------")
            print("Confusion matrix")
            print(confusion_matrix(y_test, y_pred_aud, labels=self.labels))

            print("f1 score")
            print(f1_score(y_test, y_pred_aud, labels=self.labels, average='weighted'))
            print("-----------------------------------")

        if not no_visual:
            y_pred_vis_ids = np.argmax(X_test[:, 5:], axis=1)
            y_pred_vis = [self.labels[int(i)] for i in y_pred_vis_ids]
            print("------- Visual Model ------------")
            print("Confusion matrix")
            print(confusion_matrix(y_test, y_pred_vis, labels=self.labels))

            print("f1 score")
            print(f1_score(y_test, y_pred_vis, labels=self.labels, average='weighted'))
            print("-----------------------------------")


def predict(model, video_path, debug=False):
    features = []
    if not model.no_audio:
        audio_features = model.extract_audio_prediction(video_path)
        y_pred_aud_ids = np.argmax(audio_features)
        y_pred_aud = [model.labels[y_pred_aud_ids]]
    else:
        audio_features = np.array([])
        y_pred_aud = None

    if not model.no_visual:
        video_features = model.extract_video_prediction(video_path, debug)
        y_pred_vis_ids = np.argmax(video_features)
        y_pred_vis = [model.labels[y_pred_vis_ids]]
    else:
        video_features = np.array([])
        y_pred_vis = None

    features.append(np.concatenate((audio_features, video_features)))
    pred_a, id_a = np.max(audio_features), np.argmax(audio_features)
    pred_b, id_b = np.max(video_features), np.argmax(video_features)

    return model.meta_clf.predict(features),\
           model.labels[id_a] if pred_a > pred_b else model.labels[id_b],\
           y_pred_aud, y_pred_vis, audio_features, video_features
