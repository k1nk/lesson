#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
import argparse

parser = argparse.ArgumentParser(description='This program Makes train and test data from images that are in categorized folders.')
parser.add_argument('dir', help='directory name or path that categorized folders are included')
parser.add_argument('cat', nargs='+', help='Directory names of the categories to process.Directory names are indexed from zero with the given order.')
parser.add_argument('--out', nargs='?', const='out.npy', default='out.npy', help='filename or path to output in npy format.')
args = parser.parse_args()

#root_dir = "./Pictures/"
root_dir = args.dir
#categories = ["red+apple", "green+apple"]
categories = args.cat
nb_classes = len(categories)

out_filename = args.out
image_size = 32
#image_size = 64
#image_size = 48
#image_size = 128

X = []
Y = []
for idx, cat in enumerate(categories):
    files = filter((lambda f_or_d: os.path.isfile(f_or_d)), glob.glob(root_dir + "/" + cat + "/*"))
    print("---", cat, "を処理中")
    for i, f in enumerate(files):
        img = load_img(f, target_size=(image_size,image_size))
        data = img_to_array(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.25,shuffle=True)
xy = (X_train, X_test, y_train, y_test)
#np.save("./image/apple.npy", xy)
np.save(out_filename, xy)
print("ok,", len(Y))
