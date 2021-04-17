# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:49:32 2020

@author: shankarj

visualization script for obj detection.

Code imported and modidied from 
https://github.com/yhenon/pytorch-retinanet
"""

import torch
import numpy as np
import time
import os
import csv
import cv2
import matplotlib.pyplot as plt

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption, isPred=False):
    b = np.array(box).astype(int)
    if isPred:
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 
                    2, (0, 0, 255), 4, cv2.LINE_AA)
    else:
        cv2.putText(image, caption, (b[2], b[3] + 10), cv2.FONT_HERSHEY_PLAIN, 
                    2, (255, 0, 0), 4, cv2.LINE_AA)
    #cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image_local(image_path, model, class_list, thresh=0.5):
    
    img_extensions = ['.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff']

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key   

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in os.listdir(image_path):
        
        if max([img_name.find(ext) for ext in img_extensions]) == -1:
            continue
        
        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > thresh)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imshow('detections', image_orig)
            cv2.waitKey(0)
            
    cv2.destroyAllWindows()
    
def plot_ground_truth(data_frame, col_name, num_plots, img_name_list):
    
    #if no requested images, choose random images
    if num_plots != len(img_name_list):
        img_name_list = list(data_frame[col_name].sample(num_plots))
        
    for i in range(num_plots):
        fig, ax = plt.subplots(1, 2, figsize = (10, 10))
        ax = ax.flatten()
        
        image_name = img_name_list[i]
        records = data_frame[data_frame[col_name] == image_name] 
        
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image2 = image
        
        ax[0].set_title('Original Image')
        ax[0].imshow(image)
        
        for idx, row in records.iterrows():
            box = row[['xmin', 'ymin', 'xmax', 'ymax', 'class']].values
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            label = str(box[4])
            
            cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
            draw_caption(image2, (xmin, ymin, xmax, ymax), label)
        
        ax[1].set_title('Annotated Image')
        ax[1].imshow(image2, interpolation='nearest')
    
        plt.show()

def plot_predictions(data_frame, class_frame, col_name, num_plots, model, thresh=0.5):    
   
    img_name_list = list(data_frame[col_name].sample(num_plots))
    
    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()   
        
    for i in range(num_plots):
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
        #ax = ax.flatten()
        
        image_name = img_name_list[i]
        records = data_frame[data_frame[col_name] == image_name] 
        
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if image is None:
            continue
        image_orig = image.copy()
        image_orig /= 255.0
        
        #Prepare the image for detection (pre-proc)        
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))        
        
        #plot the prediction
        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()
           
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())            
            idxs = np.where(scores.cpu() > thresh)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                #label_name = labels[int(classification[idxs[0][j]])]
                class_idx = int(classification[idxs[0][j]])
                label_name = class_frame[class_frame['id'] == class_idx][0].item()
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)               
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), (0, 0, 255), 2)           
        
        #plot the ground truth
        for idx, row in records.iterrows():
            box = row[['xmin', 'ymin', 'xmax', 'ymax', 'class']].values
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            label = str(box[4])
            
            draw_caption(image_orig, (xmin, ymin, xmax, ymax), label)
            cv2.rectangle(image_orig, (xmin, ymin), (xmax, ymax), (255,0,0), 2)            
        
        ax.set_title('Ground truth(Red) vs Prediction(Blue)')
        ax.imshow(image_orig, interpolation='nearest')
    
        plt.show()
        return scores, classification, transformed_anchors
