# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:12:35 2020

@author: shankarj
"""

import mxnet as mx
from mxnet import autograd, gluon, init, np, npx, image
from mxnet.gluon import nn
import matplotlib.pyplot as plt
npx.set_np()

style_img = image.imread('Data/style_transfer/StarryNight_res.jpg')
content_img = image.imread('Data/style_transfer/City_res.jpg')

plt.imshow(style_img.asnumpy())
plt.imshow(content_img.asnumpy())

model = gluon.model_zoo.vision.vgg19(pretrained=True)
vgg19_features = model.features
num_gpu = 1
ctx = [mx.gpu(i) for i in range(num_gpu)]

rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)

style_layers, content_layers = [0, 5, 10, 19, 28], [25]

def get_features(x, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(max(content_layers + style_layers) + 1):
        x = vgg19_features[i](x)
        if i in style_layers:
            styles.append(x)
        if i in content_layers:
            contents.append(x)
    
    return contents, styles

def get_contents(image_shape, device):
    content_x = preprocess(content_img, image_shape).copyto(device)
    content_y, _ = get_features(content_x, content_layers, style_layers)
    return content_x, content_y

def get_styles(image_shape, device):
    style_x = preprocess(style_img, image_shape).copyto(device)
    _, style_y = get_features(style_x, content_layers, style_layers)
    return style_x, style_y

def gram_matrix(x):
    _, d, h, w = x.shape
    x = x.reshape(d, h*w)
    gram = np.dot(x, x.T)/(d*h*w)
    return gram

def mse_loss(x1, x2):
    loss = np.square(x1-x2).mean()
    if loss is None:
        return 0
    else:
        return loss

content_weight = 1  #alpha
style_weight = 1e2  #beta

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [mse_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [mse_loss(gram_matrix(Y_hat), gram_matrix(Y)) * style_weight 
                for Y_hat, Y in zip(styles_Y_hat, styles_Y)]
    
    # Add up all the losses
    l = sum(styles_l + contents_l)
    return contents_l, styles_l, l

class GeneratedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(GeneratedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
    
def get_inits(X, device, lr):
    gen_img = GeneratedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})    
    return gen_img(), trainer

def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, trainer = get_inits(X, device, lr)    
    for epoch in range(1, num_epochs+1):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = get_features(
                X, content_layers, style_layers)
            contents_l, styles_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y)
        l.backward()
        trainer.step(1)
        npx.waitall()
        if epoch % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        if epoch % 100 == 0:
            print('Total loss: ', l.item())
            print('Iteration: ', epoch+1)
            plt.imshow(postprocess(X).asnumpy())
            plt.axis("off")
            plt.show()
    return X

image_shape = (content_img.shape[1], content_img.shape[0])
device = ctx[0]
vgg19_features.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.01, 3000, 200)

plt.imshow(postprocess(output).asnumpy())