#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time

import numpy as np
from PIL import Image


import six
#import six.moves.cPickle as pickle
import cPickle as pickle
from six.moves import queue

import chainer
import matplotlib.pyplot as plt
import numpy as np
import math
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from matplotlib.ticker import *
from chainer import serializers
from chainer import Reporter, report, report_scope
from chainer.training import extensions
import nin
import alex
import googlenet
import googlenetbn
import resnet50
import alex_mini
import alex_mini2
archs = {
    'alex': alex.Alex,
    'alex_fp16': alex.AlexFp16,
    'googlenet': googlenet.GoogLeNet,
    'googlenetbn': googlenetbn.GoogLeNetBN,
    'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
    'nin': nin.NIN,
    'resnet50': resnet50.ResNet50,
    'alex_mini': alex_mini.AlexMini,
    'alex_mini_fp16': alex_mini.AlexMiniFp16,
    'alex_mini2': alex_mini2.AlexMini2,
    'alex_mini2_fp16': alex_mini2.AlexMini2Fp16
}

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label

parser = argparse.ArgumentParser(
    description='model test using chainer')
#parser.add_argument('image', help='Path to inspection image file')
parser.add_argument('test', help='Path to test image index file')
parser.add_argument('--mean', default='mean.npy',help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--model','-m',default='result/model_iter_12000', help='Path to model file')
parser.add_argument('--root', '-R', default='.',
                    help='Root directory path of image files')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU')
parser.add_argument('--arch', '-a', choices=archs.keys(), default='alex_mini2',
                    help='Convnet architecture')
parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                    help='Validation minibatch size')
#parser.add_argument('--loaderjob', '-j', type=int,
#                    help='Number of parallel data loading processes')
args = parser.parse_args()
model = archs[args.arch]()
serializers.load_npz(args.model, model)
#cropwidth = 256 - model.insize
#model.to_cpu()
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
    model.to_gpu()

mean = np.load(args.mean)
test = PreprocessedDataset(args.test, args.root, mean, model.insize, False)
test_iter = chainer.iterators.SerialIterator(
    test, args.val_batchsize, repeat=False, shuffle=False)
reporter = Reporter()
reporter.add_observer('test:', model)
observation = {}
with reporter.scope(observation):
    with chainer.using_config('train', False):
        with chainer.no_backprop_mode():
            ev = extensions.Evaluator(test_iter, model, device=args.gpu)
            res = ev.evaluate()
            print(res)
