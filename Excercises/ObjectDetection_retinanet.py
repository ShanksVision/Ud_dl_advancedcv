# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:08:39 2020

@author: shankarj

"""
import torch as pt
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import sys
from retinanet import model
from visualize_detection import detect_image_local

gpu = pt.device('cuda:0')
retnet_model = model.resnet50(80)   
retnet_model.load_state_dict(pt.load('../Models/coco_resnet_50_map_0_335_state_dict.pt'))

#Get the class name csv
class_names = [l.rstrip() for l in open('data/coco_categories.txt')]

#write it in a format that our retinanet uses
f = open('data/coco_csv_classnames.csv', 'w')
for i, class_name in enumerate(class_names):
    f.write(f'{class_name},{i}\n')

f.close()

#Pas image path, model, class csv and thershold
detect_image_local('data/obj_det/', retnet_model, 'data/coco_csv_classnames.csv', 0.6)
