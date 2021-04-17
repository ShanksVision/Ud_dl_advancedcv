# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:57:49 2020

@author: shankarj

using pytorch implementation of retina net from here
https://github.com/yhenon/pytorch-retinanet

"""

import torch as pt
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import sys
import collections
from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, \
     AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet import csv_eval
import pandas as pd
import numpy as np
import cv2
from visualize_detection import plot_ground_truth, plot_predictions

#open the annotations file
dir_path = 'data/microcontroller'
train_df = pd.read_csv(f'{dir_path}/train.csv')
test_df = pd.read_csv(f'{dir_path}/test.csv')
class_df = pd.read_csv(f'{dir_path}/class.csv')

#Check class distribution
pd.plotting.hist_series(train_df['class'])
num_train_imgs = len(train_df)
num_test_imgs = len(test_df)
num_classes = len(class_df)

#plot some images to check annotation
#particular image
plot_ground_truth(train_df, 'filename', 1, 
                  ['data/microcontroller/train/IMG_20181228_103113.jpg'])
#plot random gt images
plot_ground_truth(train_df, 'filename', 2, [])

#Set up the data loaders
dataset_train = CSVDataset(train_file=f'{dir_path}/train.csv', class_list=f'{dir_path}/class.csv',
                           transform=tv.transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
                           header=True)

dataset_test = CSVDataset(train_file=f'{dir_path}/test.csv', class_list=f'{dir_path}/class.csv',
                           transform=tv.transforms.Compose([Normalizer(), Resizer()]),
                           header=True)

sampler = AspectRatioBasedSampler(dataset_train, batch_size=10, drop_last=False)
# dataloader_train = pt.utils.data.DataLoader(dataset_train, num_workers=3, 
#                                             collate_fn=collater, batch_sampler=sampler)
dataloader_train = pt.utils.data.DataLoader(dataset_train, collate_fn=collater, 
                                            batch_sampler=sampler)

#create model
retinanet = model.resnet34(num_classes=num_classes, pretrained=True)

use_gpu = True

if use_gpu:
    if pt.cuda.is_available():
        retinanet = retinanet.cuda()

# if pt.cuda.is_available():
#     retinanet = pt.nn.DataParallel(retinanet).cuda()
# else:
#     retinanet = pt.nn.DataParallel(retinanet)

#conigure traininig parameters
retinanet.training = True
optimizer = pt.optim.Adam(retinanet.parameters(), lr=1e-3)
#scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
loss_hist = collections.deque(maxlen=500)
epochs = 15
retinanet.train()
#retinanet.module.freeze_bn()
retinanet.freeze_bn()

print('Num training images: {}'.format(len(dataset_train)))

#train the network
for epoch_num in range(epochs):

    retinanet.train()
    #retinanet.module.freeze_bn()
    retinanet.freeze_bn()

    epoch_loss = []

    for iter_num, data in enumerate(dataloader_train):
        try:
            optimizer.zero_grad()

            if pt.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), 
                                                                  data['annot'].cuda().float()])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()
            pt.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num+1, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss
        except Exception as e:
            print(e)
            continue
    
    print('Evaluating dataset')
    mAP = csv_eval.evaluate(dataset_test, retinanet)
    #scheduler.step(np.mean(epoch_loss))    

print(f'Mean average precision = {mAP}')
retinanet.eval()
pt.save(retinanet, 'model_final.pt')

#plot predictions
scores, classification, transformed_anchors = plot_predictions(test_df, class_df, 
                                                               'filename', 3, retinanet, 0.1)
