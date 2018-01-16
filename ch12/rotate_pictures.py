#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# usage: ./increase_picture.py hogehoge.jpg
#

import cv2
import numpy as np
import sys
import os

def rotate_pic(src,dst):
    print(src)
    # ルックアップテーブルの生成
    # 画像の読み込み
    img_src = cv2.imread(src, 1)
    trans_img = []
    trans_img.append(img_src)

    # 反転
    flip_img = []
    for img in trans_img:
        rows,cols,ch = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        dst90 = cv2.warpAffine(img,M,(cols,rows))
        flip_img.append(dst90)

        M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        dst180 = cv2.warpAffine(img,M,(cols,rows))
        flip_img.append(dst180)

        M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
        dst270 = cv2.warpAffine(img,M,(cols,rows))
        flip_img.append(dst270)

    trans_img.extend(flip_img)

    # 保存
    #if not os.path.exists("trans_images"):
    #    os.mkdir("trans_images")

    #base =  os.path.splitext(os.path.basename(src))[0] + "_"
    dst_dir = os.path.split(dst)[0]
    base =  os.path.splitext(os.path.basename(dst))[0] + "_"
    dst_base = os.path.join(dst_dir,base)

    img_src.astype(np.float64)
    for i, img in enumerate(trans_img):
        # 比較用
        # cv2.imwrite("trans_images/" + base + str(i) + ".jpg" ,cv2.hconcat([img_src.astype(np.float64), img.astype(np.float64)]))
        #cv2.imwrite("trans_images/" + base + str(i) + ".jpg" ,img)
        dest_path = dst_base + str(i) + ".jpg"
        cv2.imwrite(dest_path ,img)

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
                rotate_pic(srcname, dstname)
        except (IOError, os.error) as why:
            print ("Can't translate %s to %s: %s" % (srcname, dstname, str(why)))

if __name__ == '__main__':
    source_dir = sys.argv[1]
    dest_dir =  source_dir + "_rotate"
    transtree(source_dir,dest_dir)
    #for source_imgpath in os.listdir(source_dir):
    #    print source_imgpath
    #    pathname = os.path.join(source_dir,source_imgpath)
    #    increase_pic(pathname)
        #pwd = os.path.abspath(".")
