# -*- coding: utf-8 -*-
#import apple_keras as apple
import train_keras as tr
from train_keras import build_model
import sys, os
import numpy as np
import subprocess
import cv2
from keras.preprocessing.image import load_img, img_to_array
import jtalk

cam = cv2.VideoCapture(0)
image_size = 32
#categories = [u"赤りんご", u"青りんご"]
#categories = ["apple", "BOSS","GEORGIA","UCC","WONDA"]
categories = [u"りんご", u"ボス",u"ジョージア",u"ユーシーシー",u"ワンダ"]
nb_classes = len(categories)

def main():
    while(True):
        ret, frame = cam.read()
        cv2.imshow("Show FLAME Image", frame)

        k = cv2.waitKey(1)
        if k == ord('s'):
            cv2.imwrite("output.png", frame)
            cv2.imread("output.png")

            X = []
            img = load_img("./output.png", target_size=(image_size,image_size))
            in_data = img_to_array(img)
            X.append(in_data)
            X = np.array(X)
            X  = X.astype("float")  / 256

            #model = apple.build_model(X.shape[1:])
            #model = tr.build_model(X.shape[1:], nb_classes)
            model = build_model(X.shape[1:], nb_classes)
            #model.load_weights("./image/apple-model.h5")
            model.load_weights("model.h5")

            pre = model.predict(X)
            print(pre)
            for i in range(len(categories)):
                if pre[0][i] > 0.5:
                    print(categories[i])
                    text = u'これは' + categories[i]+ u'だよ'
                    text = text.encode('utf-8')
                    jtalk.jtalk(text)

        elif k == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
