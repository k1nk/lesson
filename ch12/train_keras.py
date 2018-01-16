# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import argparse
from keras.callbacks import ModelCheckpoint
#import matplotlib.pyplot as plt

#root_dir = "./image/"
#categories = ["red+apple", "green+apple"]
#nb_classes = len(categories)
#image_size = 32

def main():
    args = get_args()

    nb_classes = args.nb_classes
    input_file = args.input
    model_output_hdf5_file = args.output
    n_epoch = args.epoch

    #X_train, X_test, y_train, y_test = np.load("./image/apple.npy")
    X_train, X_test, y_train, y_test = np.load(input_file)
    X_train = X_train.astype("float") / 256
    X_test  = X_test.astype("float")  / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)
    model = model_train(X_train, y_train, nb_classes, model_output_hdf5_file,n_epoch)
    model_eval(model, X_test, y_test)
    #backend.clear_session()

def build_model(in_shape,n_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def model_train(X, y,n_classes, output_hdf5_file, nb_epoch):
    model = build_model(X.shape[1:],n_classes)
    model.summary()
    history = model.fit(X, y, batch_size=32, nb_epoch=nb_epoch, validation_split=0.1,
    callbacks=[ModelCheckpoint(output_hdf5_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])
    #hdf5_file = "./image/apple-model.h5"
    #hdf5_file = model_output_file
    model.save_weights("final_"+output_hdf5_file)
    print(history.history)
    #plot_history(history)
    return model

#def plot_history(history):
#    plt.plot(history.history['acc'],"o-",label="accuracy")
#    plt.plot(history.history['val_acc'],"o-",label="val_acc")
#    plt.title('model accuracy')
#    plt.xlabel('epoch')
#    plt.ylabel('accuracy')
#    plt.ylim(0, 1)
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    plt.show()
#
#    plt.plot(history.history['loss'],"o-",label="loss",)
#    plt.plot(history.history['val_loss'],"o-",label="val_loss")
#    plt.title('model loss')
#    plt.xlabel('epoch')
#    plt.ylabel('loss')
#    plt.ylim(ymin=0)
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    plt.show()

def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])

def get_args():
    parser = argparse.ArgumentParser(description='This program Makes train and test data from images that are in categorized folders.')
    parser.add_argument('nb_classes', type=int, help='Number of categories')
    parser.add_argument('--input', nargs='?', const='out.npy', default='out.npy', help='filename or path to input in npy format.')
    parser.add_argument('--output', nargs='?', const='model.h5', default='model.h5', help='filename or path of the model to output in h5 format.')
    parser.add_argument('--epoch', nargs='?', const=10, default=10, type=int, help='number of epoch to train.')

    return parser.parse_args()

if __name__ == "__main__":
    main()
