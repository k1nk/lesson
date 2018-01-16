#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# usage: ./increase_picture.py hogehoge.jpg
#
import sys
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def rotate_pic(srcfile,dstdir):
    print(srcfile)
    img = load_img(srcfile)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
     #the .flow() command below generates batches of randomly transformed images
     #and saves the results to the `preview/` directory
    base =  os.path.splitext(os.path.basename(srcfile))[0]
    g = datagen.flow(x, batch_size=1,save_to_dir=dstdir, save_prefix=base, save_format='jpeg')
    for i in range(20):
        batch = g.next()
    #i = 0
    #for batch in datagen.flow(x, batch_size=1,
    #                          save_to_dir=dstdir, save_prefix='a', save_format='jpeg'):
    #    i += 1
    #    if i > 20:
    #        break  # otherwise the generator would loop indefinitely

def transtree(src, dst):
    names = os.listdir(src)
    if not os.path.exists(dst):
        os.mkdir(dst)
    for name in names:
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            #if os.path.islink(srcname):
                #linkto = os.readlink(srcname)
                #os.symlink(linkto, dstname)
            if os.path.isdir(srcname):
                transtree(srcname, dstname)
            else:
                rotate_pic(srcname, dst)
                #rotate_pic(srcname, dstname)
        except (IOError, os.error) as why:
            print ("Can't translate %s to %s: %s" % (srcname, dstname, str(why)))

if __name__ == '__main__':
    source_dir = sys.argv[1]
    dest_dir =  source_dir + "_rotate3"
    transtree(source_dir,dest_dir)
    #for source_imgpath in os.listdir(source_dir):
    #    print source_imgpath
    #    pathname = os.path.join(source_dir,source_imgpath)
    #    increase_pic(pathname)
        #pwd = os.path.abspath(".")
