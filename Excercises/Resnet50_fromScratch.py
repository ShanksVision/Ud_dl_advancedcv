# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:07:21 2020

@author: shankarj
"""
import mxnet as mx
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import time
import matplotlib.pyplot as plt
import sklearn.metrics as met
import sys

def tensor_to_image(tensor):
    image = np.movaxis(tensor, 0, -1)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image.asnumpy()

npx.set_np()

def plot_loss_wrong_preds(history, x=None, y=None, yhat=None, labels=None):
    #Plot the train loss and val loss
    plt.plot(history[0], label='train_err')
    plt.plot(history[1], label='val_err')
    plt.title('Error trend')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()    
   
    #Show some misclassified samples    
    if(labels):       
        #get some wrong predictions
        mis_idx = np.where(y != yhat)[0]
        #get some correct predictions
        idx = np.where(y == yhat)[0]        
        wrong_preds = np.random.choice(mis_idx, size=min(len(mis_idx), 8)) 
        correct_preds = np.random.choice(idx, size=min(len(idx), 8))
        ax = []
        fig=plt.figure(figsize=(12, 12))
        columns = 4
        rows = 4
        
        for i, j in enumerate(correct_preds):
            j = j.item()
            if(type(x) is np.ndarray):
                img = x[j]
            else:
                img = plt.imread(x[j])
            ax.append(fig.add_subplot(rows, columns, i+1))
            ax[-1].set_title(f'true: {labels[y[j].item()]}, pred: {labels[yhat[j].item()]}',
                             color='g')                             
            plt.imshow(img)
        for i, j in enumerate(wrong_preds):
            j = j.item()
            if(type(x) is np.ndarray):
                img = x[j]
            else:
                img = plt.imread(x[j])
            ax.append(fig.add_subplot(rows, columns, i+9))
            ax[-1].set_title(f'true: {labels[y[j].item()]}, pred: {labels[yhat[j].item()]}',
                             color='r')                             
            plt.imshow(img)
        plt.tight_layout(pad=1.2)    
        fig.suptitle('Sample Correct(8) & Wrong predictions(8)', y = 0.001)
        plt.show()
    
def print_model_metrics(y, yhat, labels, title, stream=sys.stdout, wrong_preds=False):
    #Check if y is one hot encoded
    if(len(y.shape) != 1):
        y = y.argmax(axis=1)
        yhat = yhat.argmax(axis=1)
        
    print('\n' + title + '\n------------\n', file=stream)   
    
    print("Classification Metrics\n----------\n", file=stream)  
    print(met.classification_report(y, yhat, target_names=labels,
                                    zero_division=1), file=stream) 
    
    print("Confusion Matrix\n----------\n", file=stream)
    print(met.confusion_matrix(y, yhat), file=stream)    
        
    if(wrong_preds):
        print("Wrong Predictions\n----------\n", file=stream)
        mis_idx = np.where(y != yhat)[0]
        size = min(len(mis_idx), 10)
        wrong_preds = np.random.choice(mis_idx, size=size)
        for i in range(size):
            #print(df_test.iloc[wrong_preds[i]], file=stream)
            print('Original Label : {0}'.format(y[wrong_preds[i]]), 
                  file=stream)
            print('Predicted Label : {0}'.format(yhat[wrong_preds[i]]),
                  file=stream)
            print('********************', file=stream)


class ResidualBlock(nn.Block):
    def __init__(self, in_channels, num_channels, is_convblock, strides):
        super().__init__()
        self.conv1 = nn.Conv2D(num_channels[0], kernel_size=1, strides=strides, 
                               in_channels=in_channels)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(num_channels[1], kernel_size=3, padding=1, 
                               use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(num_channels[2], kernel_size=1)
        self.bn3 = nn.BatchNorm()
        if is_convblock:
            self.conv_skip = nn.Conv2D(num_channels[2], kernel_size=1,  
                               strides=strides, use_bias=False,  in_channels=in_channels)
            self.bn_skip = nn.BatchNorm()
        else:
            self.conv_skip = None
            self.bn_skip = None
            
    def forward(self, x):   
        y = self.conv1(x)
        y = self.bn1(y)
        y = npx.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = npx.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        
        if self.conv_skip:
            x = self.conv_skip(x)
            x = self.bn_skip(x)
        
        return npx.relu(x + y)

def resnet_layers(num_blocks, in_channels, num_channels, is_convblock, stride):
    resnet_blk = nn.Sequential()
    for i in range(num_blocks):
        resnet_blk.add(ResidualBlock(in_channels, num_channels, is_convblock, stride))
        
    return resnet_blk

model = nn.Sequential()
model.add(nn.Conv2D(in_channels = 3, channels=64, kernel_size=7, padding=3, strides=2))
model.add(nn.BatchNorm())
model.add(nn.Activation('relu'))
model.add(nn.MaxPool2D(pool_size=3, padding=1, strides=2))
model.add(resnet_layers(1, 64, [64, 64, 256], True, 1))
model.add(resnet_layers(2, 256, [64, 64, 256], False, 1))
model.add(resnet_layers(1, 256, [128, 128, 512], True, 2))
model.add(resnet_layers(3, 512, [128, 128, 512], False, 1))
model.add(resnet_layers(1, 512, [256, 256, 1024], True, 2))
model.add(resnet_layers(5, 1024, [256, 256, 1024], False, 1))
model.add(resnet_layers(1, 1024, [512, 512, 2048], True, 2))
model.add(resnet_layers(2, 2048, [512, 512, 2048], False, 1))
model.add(nn.GlobalAvgPool2D())
model.add(nn.Dense(12))
#model.initialize(mx.init.Xavier())
    
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
transform_train = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5],
                                                          [0.5, 0.5, 0.5])])

train_folder = gluon.data.vision.ImageFolderDataset('data/fruits/train')                                                    
test_folder = gluon.data.vision.ImageFolderDataset('data/fruits/test')
train_data = gluon.data.DataLoader(train_folder.transform_first(transform_train), 
                                   batch_size = batch_size, shuffle=True)
test_data = gluon.data.DataLoader(test_folder.transform_first(transform_test),
                                  batch_size = batch_size, shuffle=False)

classes = train_folder.synsets
#Data representation and getting path
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
epochs=20
terr_history=[]
verr_history=[]

def test(ctx, val_data, y_pred=None, y_test=None):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx)
        labels = gluon.utils.split_and_load(batch[1],ctx)
        samples = len(data[0])
        outputs = [model(X) for X in data]
        
        if y_pred is not None:
            y_pred[i*samples:(i*samples)+samples] = outputs[0].argmax(axis=1)
            y_test[i*samples:(i*samples)+samples] = labels[0]
        else:
            metric.update([labels[0].as_nd_ndarray()], [outputs[0].as_nd_ndarray()])
    if y_pred is not None:
        return y_pred, y_test
    else:
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


#print('[Finished] Test-acc: %.3f' % (test_acc))

#Final predictions
y_test = np.zeros(len(test_folder), dtype=np.int32, ctx=ctx[0])
y_pred = np.zeros(len(test_folder), dtype=np.int32, ctx=ctx[0])
y_pred, y_test = test(ctx, test_data, y_pred, y_test)

#plot loss curves and missed predictions
x_test = [x[0] for x in test_folder.items]
model_history = [terr_history, verr_history]
plot_loss_wrong_preds(model_history, x_test, y_test, y_pred, classes)

#save metrics
file_stream = open('results/fruits360_resnet50.txt', 'w')
print("Model Summary\n----------\n", file=file_stream)  
print(model.summary, file=file_stream)
#print_model_metrics(y_train, y_pred_train, class_labels, 'Train Metrics', file_stream)
print_model_metrics(y_test.asnumpy(), y_pred.asnumpy(), classes, 'Test Metrics', file_stream)
file_stream.close()




