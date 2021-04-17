# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:26:14 2021

@author: shankarj
"""

import torch
import torch.nn.functional as F
import torch.optim as optimizer

class UNetEncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UNetEncoderBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x
    
class UNetDecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UNetDecoderBlock, self).__init__()
        self.tconv1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,
                                               stride=2)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        
    def forward(self, x1, x2):
        #Transpose conv
        x1 = self.tconv1(x1)
        
        #concat features from encoder
        ydiff = x2.size()[2] - x1.size()[2]
        xdiff = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [xdiff // 2, xdiff - xdiff // 2, 
                        ydiff // 2, ydiff - ydiff // 2])
        x = torch.cat([x2, x1], dim=1)
        
        #convolve for decoder
        x = F.relu(self.conv1(x))
        x = self.bn1(x)        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        
        return x
    
class UNetFinalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(UNetFinalBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size)      
                                              
        
    def forward(self, x):      
        x = torch.sigmoid(self.conv1(x))       
        return x 
    
class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UNet, self).__init__()
        self.encode1 = UNetEncoderBlock(in_channels, 64, kernel_size)
        self.encode2 = UNetEncoderBlock(64, 128, kernel_size)
        self.encode3 = UNetEncoderBlock(128, 256, kernel_size)
        self.encode4 = UNetEncoderBlock(256, 512, kernel_size)        
        self.encode5= UNetEncoderBlock(512, 1024, kernel_size)
        
        self.decode4 = UNetDecoderBlock(1024, 512, kernel_size)
        self.decode3 = UNetDecoderBlock(512, 256, kernel_size)
        self.decode2 = UNetDecoderBlock(256, 128, kernel_size)        
        self.decode1 = UNetDecoderBlock(128, 64, kernel_size)
        
        self.output = UNetFinalBlock(64, out_channels)
   
    #buggy, not used
    def encoder_to_decoder_bypass(self, dec_op, enc_op):
        crop = enc_op.size()[2] - dec_op.size()[2] // 2
        enc_op = F.pad(enc_op, (-crop, -crop, -crop, -crop))
        return torch.cat((dec_op, enc_op), 1)
        
    def forward(self, x):
        x1 = self.encode1(x)
        x = F.max_pool2d(x1, 2)
        x2 = self.encode2(x)
        x = F.max_pool2d(x2, 2)
        x3 = self.encode3(x)
        x = F.max_pool2d(x3, 2)
        x4 = self.encode4(x)
        x = F.max_pool2d(x4, 2)
        x = self.encode5(x)
        x = self.decode4(x, x4)
        x = self.decode3(x, x3)
        x = self.decode2(x, x2)
        x = self.decode1(x, x1)
        x = self.output(x)
        
        return x
    
        