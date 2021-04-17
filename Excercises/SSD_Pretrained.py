# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:46:14 2020

@author: Shankarj

mxnet has some issues with loading pre trained nets, dropping this excercise
"""

import mxnet as mx
from gluoncv import model_zoo, data, utils
import matplotlib.pyplot as plt
from mxnet import gluon, np, npx
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

npx.set_np()
ctx = mx.gpu(0)
model = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained = True, ctx=ctx)

model = model_zoo.ssd_512_resnet18_v1_coco(pretrained=True)

