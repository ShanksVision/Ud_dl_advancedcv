# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:05:40 2021

@author: shankarj
"""

import torch
import torchvision as tv
import torch.optim as optimizer
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import numpy as np
import glob
import PIL
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import dataloader
from torch.utils.data.dataset import Dataset
from unet import UNet 

def plot_train_data(train_dataset, blend=False):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('train image')
    ax[1].set_title('mask image')
    count = len(train_dataset)
    idx = np.random.choice(count)
    im, mask = train_dataset[idx]
    im = im.numpy()[0];
    mask = mask.numpy()[0]    
    if blend:
        mask = np.multiply(im, mask)
        # mask = Image.fromarray(mask.astype(np.uint8))
        # im = Image.fromarray(im.astype(np.uint8))
        # mask = (Image.blend(im, mask, alpha=0.7))  
    ax[0].imshow(im)
    ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show() 

def plot_val_data(val_dataset, model, gpu, blend=False):
    model.to(gpu)
    model.eval()
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('test image')
    ax[1].set_title('mask image')
    ax[2].set_title('pred image')
    count = len(val_dataset)
    idx = np.random.choice(count)
    im, mask = val_dataset[idx]
    im_eval = im.unsqueeze(0).to(gpu)    
    pred = model(im_eval)
    pred = pred.squeeze(0).squeeze(0)
    pred = pred.cpu().detach().numpy()
    im = im.numpy()[0];
    mask = mask.numpy()[0]    
    if blend:
        mask = np.multiply(im, mask)
        
        # mask = Image.fromarray(mask.astype(np.uint8))
        # im = Image.fromarray(im.astype(np.uint8))
        # mask = (Image.blend(im, mask, alpha=0.7))  
    ax[0].imshow(im)
    ax[1].imshow(mask)
    ax[2].imshow(pred)
    plt.xticks([]), plt.yticks([])
    plt.show()  
    

class ImageSegDataset(Dataset):
    def __init__(self, img_dir, label_dir, scale=1):
        self.image_folder = img_dir
        self.mask_folder = label_dir
        self.scale = scale
        self.img_files = glob.glob(f'{self.image_folder}/*.*')
        self.mask_files = glob.glob(f'{self.mask_folder}/*.*') 
        
    def __len__(self):         
        return len(self.img_files)
    
    def preprocess(self, image, mask):
        width, height = image.size
        scaled_w = int(width * self.scale)
        scaled_h = int(height * self.scale)
        image = tv.transforms.Resize((scaled_h, scaled_w))(image)
        mask = tv.transforms.Resize((scaled_h, scaled_w))(mask)
        #image = image.resize((scaled_h, scaled_w), PIL.Image.NEAREST)
        #mask = mask.resize((scaled_h, scaled_w), PIL.Image.NEAREST)        
        return TVF.to_tensor(image), TVF.to_tensor(mask)
    
    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        mask = Image.open(self.mask_files[index])
        X, y = self.preprocess(img, mask)
        return X, y

#set the computing device
device = torch.device("cuda:1")

#set the training variables
batch_size = 2
epochs = 40
learning_rate = 0.0001
img_path = '../Data/xray/train/image_c'
label_path = '../Data/xray/train/label_c'    
train_db = ImageSegDataset(img_path, label_path, 0.5)
train_loader = dataloader.DataLoader(train_db, batch_size=batch_size, shuffle=False,
                                      pin_memory=True)

img_path = '../Data/xray/test/image_c'
label_path = '../Data/xray/test/label_c'    
val_db = ImageSegDataset(img_path, label_path, 0.5)
val_loader = dataloader.DataLoader(val_db, batch_size=batch_size, shuffle=False,
                                    pin_memory=True)
#View some input data
# response = 'y'
# while(response == 'y'):
#     plot_train_data(train_db, True)
#     response = input("Do you want to review another train image ? y/n : ")

#Create model and train
model = UNet(in_channels = 1, out_channels = 1)
model.to(device) 
opt = optimizer.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCELoss()    


running_loss = 0.0
val_loss = 0.0
tloss_history = []
vloss_history = []

for epoch in range(epochs):
    #train
    running_loss = 0.0
    model.train(True)
    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        opt.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        
        loss.backward()
        opt.step()
        running_loss += loss.item()
    
    print("Train loss for epoch " + str(epoch) + ":  " + str(running_loss))
    tloss_history.append(running_loss)
    
    #eval
    model.train(False)
    val_loss = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(val_loader):
             X = X.to(device)
             y = y.to(device)                     
             pred_val = model(X)
             loss = criterion(pred_val, y)
             val_loss += loss.item()
            
    print("Val loss for epoch " + str(epoch) + ":  " + str(val_loss))
    vloss_history.append(val_loss)        
    

#display the loss curves
plt.plot(tloss_history, label='train_loss')
plt.plot(vloss_history, label='val_loss')
plt.title('Loss trend')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#view some predicted data
response = 'y'
while(response == 'y'):
    plot_val_data(val_db, model, device, True)
    response = input("Do you want to review another train image ? y/n : ")