#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# usage: ./crop_pictures.py source_dir
#

import cv2
import numpy as np
import sys
import os

def crop_pic(src,dst):
    # 画像の読み込み
    img = cv2.imread(src, 1)
    img.astype(np.float64)
    target_shape = (256, 256)
    output_side_length=256
    #print src
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    resized_img = cv2.resize(img, (new_width, new_height))
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
width_offset:width_offset + output_side_length]
    cv2.imwrite(dst, cropped_img)

def is_acceptable_ext(ext):
    cv2_acceptable_extensions = [".jpg",".jpeg",".jpe",".jp2",".png",".webp",".bmp",".pbm",".pgm",".ppm",
".pxm", ".pnm", ".sr", ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic", ".dib"]
    return ext.lower() in cv2_acceptable_extensions

def is_acceptable_file(src):
    root, ext = os.path.splitext(src)
    return is_acceptable_ext(ext)

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
                if not is_acceptable_file(srcname):
                    continue
                print "going to crop %s" % (`srcname`)
                crop_pic(srcname, dstname)
        except (IOError, os.error) as why:
            print "Can't translate %s to %s: %s" % (`srcname`, `dstname`, str(why))

if __name__ == '__main__':
    source_dir = sys.argv[1]
    dest_dir =  source_dir + "_cropped"
    transtree(source_dir,dest_dir)
    #for source_imgpath in os.listdir(source_dir):
    #    print source_imgpath
    #    pathname = os.path.join(source_dir,source_imgpath)
    #    increase_pic(pathname)
        #pwd = os.path.abspath(".")
