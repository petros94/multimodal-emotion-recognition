import torch
import sys, getopt, os
import re
import warnings
warnings.filterwarnings("ignore")
import pickle
from combined_model import predict
import time

def main(argv):
    ## Read args from user

    vid_path = argv[0]

    # Load pretrained model
    model_path = 'models/combined_model.pkl'
    with open(model_path, 'rb') as f:
        test_model = pickle.load(f)

    # Make prediction
    start_time = time.time()
    output = predict(test_model, vid_path)

    print(f"Inference time: {time.time() - start_time}")
    print(f"Combined prediction: {output[0][0]}")
    print(f"Audio prediction: {output[2]}")
    print(f"Visual prediction: {output[3]}")
    print("-------------")
    print(f"Labels: {['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']}")
    print(f"Base posteriors - audio: {output[-2]}")
    print(f"Base posteriors - video: {output[-1]}")


if __name__ == "__main__":
    main(sys.argv[1:])