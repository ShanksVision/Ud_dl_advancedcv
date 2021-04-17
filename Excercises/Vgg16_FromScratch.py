# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:22:13 2020

@author: shankarj
"""
import mxnet as mx
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import time
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
    image = np.movaxis(tensor, 0, -1)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image.asnumpy()

npx.set_np()

model = nn.Sequential()
model.add(nn.Conv2D(in_channels = 3, channels=64, kernel_size=3, padding=1, activation='relu'))
model.add(nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu'))
model.add(nn.MaxPool2D(pool_size=2, strides=2))
model.add(nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu'))
model.add(nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu'))
model.add(nn.MaxPool2D(pool_size=2, strides=2))
model.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'))
model.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'))
model.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'))
model.add(nn.MaxPool2D(pool_size=2, strides=2))
model.add(nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'))
model.add(nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'))
model.add(nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'))
model.add(nn.MaxPool2D(pool_size=2, strides=2))
model.add(nn.Dense(2048, activation='relu'))
model.add(nn.Dense(1024, activation='relu'))
model.add(nn.Dense(12))
    
#get the model initialized
batch_size = 50
num_gpu = 1
ctx = [mx.gpu(i) for i in range(num_gpu)]

model.initialize(mx.init.Xavier(), ctx = ctx)

#print model data flow
#x = np.random.uniform(size=(1, 1, 100, 100))
x = np.random.uniform(size=(1, 3, 100, 100), ctx=mx.gpu(0))
for layer in model:
    x = layer(x)
    print(f'Layer : {layer.name}, output shape : {x.shape}')

#set up data augmentation, transforms and data loaders
transform_train = transforms.Compose([transforms.RandomBrightness(0.2),
                                      transforms.RandomFlipLeftRight(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5],
                                                          [0.5, 0.5, 0.5])])

train_folder = gluon.data.vision.ImageFolderDataset('data/fruits/train').transform_first(transform_train)                                                    
test_folder = gluon.data.vision.ImageFolderDataset('data/fruits/test').transform_first(transform_test)
train_data = gluon.data.DataLoader(train_folder, batch_size = batch_size, shuffle=True)
test_data = gluon.data.DataLoader(test_folder, batch_size = batch_size, shuffle=False)

# flder = gluon.data.vision.ImageFolderDataset('data/fruits/test')
# dt, lb = flder[123]
# flder.synsets[lb]
# flder.items
# flder.synsets

#set up the trainer
optimizer = mx.optimizer.Adam()
trainer = gluon.Trainer(model.collect_params(), optimizer)
metric = mx.metric.Accuracy()
loss = gluon.loss.SoftmaxCrossEntropyLoss()

#train the nn
epochs=30
terr_history=[]
verr_history=[]

def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx)
        labels = gluon.utils.split_and_load(batch[1],ctx)
        outputs = [model(X) for X in data]
        metric.update([labels[0].as_nd_ndarray()], [outputs[0].as_nd_ndarray()])
    return metric.get()

for epoch in range(epochs):
    tic = time.time()
    metric.reset()
    train_loss = 0
    
    for i, batch in enumerate(train_data):
        #get data and labels
        data = gluon.utils.split_and_load(batch[0], ctx)
        labels = gluon.utils.split_and_load(batch[1], ctx)
        
        #Autograd
        with autograd.record():
            output = [model(x) for x in data]
            loss_val = [loss(yhat, y) for yhat, y in zip(output, labels)]
            
        #Backprop
        for l in loss_val:
            l.backward()
        
        #optimize
        trainer.step(batch_size)
        
        #update metrics
        train_loss += sum([l.sum().item() for l in loss_val])
        metric.update([labels[0].as_nd_ndarray()], [output[0].as_nd_ndarray()])
        
    name, acc = metric.get()
    #get metric on test data
    name, val_acc = test(ctx, test_data)
    
    #update 
    terr_history.append(1-acc)
    verr_history.append(1-val_acc)
        
    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
             (epoch, acc, train_loss/len(batch), val_acc, time.time() - tic))

_, test_acc = test(ctx, test_data)
print('[Finished] Test-acc: %.3f' % (test_acc))

plt.plot(terr_history, label='train_err')
plt.plot(verr_history, label='val_err')
plt.title('Err trend')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


