import train_keras as train_keras
#import train_keras_alex as train_keras
#import sys, os
import numpy as np
#import subprocess
#import cv2
from keras.preprocessing.image import load_img, img_to_array
import argparse
from keras.backend import tensorflow_backend as backend

parser = argparse.ArgumentParser(description='This program predict categories from the input image.')
parser.add_argument('image', help='filename or path of the input image file.')
parser.add_argument('--model', nargs='?', const='model.h5', default='model.h5', help='filename or path of the model in h5 format.')

args = parser.parse_args()

input_image_file = args.image
#nb_classes = args.nb_classes
model_file = args.model

image_size = 32
#image_size = 64
#image_size = 48
#image_size = 128

categories = ["apple", "BOSS","GEORGIA","UCC","WONDA"]
nb_classes = len(categories)

def main():
    X = []
    img = load_img(input_image_file, target_size=(image_size,image_size))
    in_data = img_to_array(img)
    X.append(in_data)
    X = np.array(X)
    X  = X.astype("float")  / 256

    model = train_keras.build_model(X.shape[1:],nb_classes)
    #model.load_weights("./image/apple-model.h5")
    model.load_weights(model_file)

    pre = model.predict(X)
    print(pre)
    for i in range(len(categories)):
        if pre[0][i] > 0.5:
            print(categories[i])

    backend.clear_session()

if __name__ == '__main__':
    main()
