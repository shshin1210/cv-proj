import datasets
#from dataset import dataset
from model_resnet import resnet34
from model import VGG

import sklearn
from sklearn.model_selection import KFold

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.image as img

import torch
from torch import batch_norm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib.image as img

import cv2
import numpy as np

import os



os.environ['CUDA_VISIBLE_DEVICES'] = "7"

print('device 개수:', torch.cuda.device_count())
print('현재 device:', torch.cuda.current_device())


# bring data
dataset_nl = datasets.C100Dataset('./dataset/data/cifar100_nl.csv')
#dataset_nl = dataset.C100Dataset()
[tr_x, tr_y, ts_x] = dataset_nl.getDataset() # trainx, trainy, testx

# wonl test set
dataset_test = datasets.C100Testset('./dataset/data/cifar100_nl_test.csv')
[finaltest_x, finaltest_y] = dataset_test.getDataset()


#label list 100개
label = set(tr_y) 
label = np.array(list(label)) # 100개의 label list


file_dir = './dataset/'
alp = 0.2
K = len(label)

# train & val & test
trainx = np.array([img.imread(file_dir+u).T for i,u in enumerate(tr_x)]) #trainx filename
trainy = np.ones([len(tr_y), 100], dtype='int32')*(alp/K) # trainy 9999x100 np, 0.002

# u==v / change to int
for i,u in enumerate(label):
    for j,v in enumerate(tr_y):
        if v==u: trainy[j][i] += 1-alp

# test filename
testx = np.array([img.imread(file_dir+u).T for i,u in enumerate(finaltest_x)])
testy = np.ones([len(finaltest_y), 100], dtype='int32')*(alp/K) # trainy 9999x100 np, 0.002
for i,u in enumerate(label):
    for j,v in enumerate(finaltest_y):
        if v==u: testy[j][i] += 1-alp

# datas to tensor
trainx, testx = torch.FloatTensor(trainx), torch.FloatTensor(testx)
trainy = torch.FloatTensor(trainy)
testy = torch.FloatTensor(testy)

# dataset
dataset = []
for i in range(len(trainx)):
    dataset.append([trainx[i],trainy[i]])

# dataset_test
dataset_t = []
for i in range(len(testx)):
    dataset.append([testx[i],testy[i]])
r_testloader = data.DataLoader(dataset_t,batch_size= 100, shuffle = True, num_workers=2)


# loss function
def loss2(output, y):
    #return F.binary_cross_entropy(output, y)
    
    #return nn.CrossEntropyLoss(output,y)
    return torch.nn.functional.mse_loss(output, y)

# accuracy
def accuracy(output, y):
    return torch.sum(torch.argmax(output, axis=1)==torch.argmax(y, axis=1))

print("===> building model")
# model
model = resnet34()
#model = VGG()
if torch.cuda.is_available():
    model = model.cuda()

#optimizer
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
#n_epochs = 50
n_epochs = 3

# loss & acc
tr_loss, vl_loss, test_loss = np.zeros(n_epochs, dtype='float64'), np.zeros(n_epochs, dtype='float64')
tr_accy, vl_accy, ts_accy = np.zeros_like(tr_loss), np.zeros_like(tr_loss), np.zeros_like(tr_loss)


kfold = KFold(n_splits = 4, random_state = 0, shuffle = True)
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print("FOLD", fold)
    print('==================================================')

    print("train index:", train_idx, "validate index:", val_idx)
    X_train, X_val = trainx[train_idx], trainx[val_idx]
    y_train, y_val = trainy[train_idx], trainy[val_idx]
    
    #train_data loader
    train_data = []
    x_tr_list = list(X_train)
    y_tr_list = list(y_train)
    for i in range(len(X_train)):
        train_data.append([x_tr_list[i], y_tr_list[i]])
    trainloader = data.DataLoader(train_data, batch_size=128, shuffle = True, num_workers=2)

    #test data loader

    # test data loader
    test_data = []
    x_vl_list = list(X_val)
    y_vl_list = list(y_val)
    for i in range(len(X_val)):
        test_data.append([x_vl_list[i], y_vl_list[i]])
    testloader = data.DataLoader(test_data, batch_size= 100, shuffle = True, num_workers=2)


    # Init NN
    model = resnet34()
    model = model.cuda()

    # Init optimizer
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # Run Training loop for num epochs

    # Training
    print('Training\n')
    for epoch in range(n_epochs):
        print('Epoch:', epoch)
        losst = 0.
        accyt = 0
        lossv = 0.
        accyv = 0
        total = 0
        correct = 0
        total_v = 0
        correct_v = 0

        cnt = 0
        cnt2 = 0
        model.train()
        for batchnum, data in enumerate(trainloader, 0):
            # get inputs
            img, tar = data
            img, tar = img.cuda(), tar.cuda()
            output = model(img)
            loss = loss2(output, tar)
            accy = accuracy(output, tar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losst += loss.item()
            accyt += accy
            cnt +=1
            
            if batchnum == 292:
                print('Loss after mini-batch %5d: %.3f' %(batchnum+1, losst/(cnt-1)))
                losst = 0.
    print('Training process finished')

    # Testing
    print('Testing\n')

    # saving model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # Evaluation for this fold
    with torch.no_grad():

        model.eval()
        for batchnum, data in enumerate(testloader,0):
            img_val, tar_val = data
            img_val, tar_val = img_val.cuda(), tar_val.cuda()
            output2 = model(img_val)
            loss = loss2(output2, tar_val)
            accy = accuracy(output2, tar_val)
            lossv += loss.item()
            accyv += accy
            cnt2 += 1

        print(cnt2)
        tr_loss[epoch] = losst
        tr_accy[epoch] = accyt
        vl_loss[epoch] = lossv
        vl_accy[epoch] = accyv

        print("Epoch:", epoch, "Train Loss:", losst, "Validation Loss:", vl_loss[epoch], \
            "Train Accuracy:", tr_accy[epoch]/(cnt-1), "Validation Accuracy:",vl_accy[epoch]/(cnt2-1))


mpl.rcParams.update({
    'font.family' : 'STIXGeneral',
    'mathtext.fontset' : 'stix',
    'xtick.direction' : 'in' ,
    'xtick.labelsize' : 13 ,
    'xtick.top' : False ,
    'ytick.direction' : 'in' ,
    'ytick.labelsize' : 13 ,
    'ytick.right' : False ,
    'axes.labelsize' : 16,
    'legend.frameon' : False,
    'legend.fontsize' : 13,
    'legend.handlelength' : 1.5,
    'savefig.dpi' : 600, 
    'savefig.bbox' : 'tight'
})

fig, ax = plt.subplots(1,2, figsize=(12,4))

ax[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))

ax[0].plot(tr_loss/tr_loss[0], '.-', label='training loss')
ax[0].plot(vl_loss/vl_loss[0], '.-', label='heldout loss')
ax[0].legend()
ax[0].set_ylim(-0.1, 1.1)
ax[0].set_xlim(0, 9)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')

ax[1].plot(tr_accy/400, '.-', label='training accuracy')
ax[1].plot(vl_accy/100, '.-', label='heldout accuracy')
ax[1].set_ylim(0, 100)
ax[1].set_xlim(0, 9)
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].legend()


plt.savefig('vgg.png')

for epoch in range(n_epochs):
    test_loss = 0
    loss_test=0
    accy_test=0
    cnt3=0
    with torch.no_grad():
        model.eval()
        for batchnum, data in enumerate(r_testloader,0):
            img_test, tar_test = data
            img_test, tar_test = img_test.cuda(), tar_test.cuda()
            output3 = model(img_test)
            loss = loss2(output3, tar_test)
            accy = accuracy(output3, tar_test)
            loss_test += loss.item()
            accy_test += accy
            cnt3 += 1

        test_loss[epoch] = loss_test
        ts_accy[epoch] = accy_test

        print("Epoch:", epoch, "Test Loss:", loss_test, \
            "Test Accuracy:", ts_accy[epoch]/(cnt-1))

# epochs = np.arange(0,200)
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1) 
# plt.xlabel('Epoch') 
# plt.ylabel('Loss') 
# plt.plot(epochs,loss_list) 
# plt.subplot(1,2,2) 
# plt.xlabel('Epoch') 
# plt.ylabel('Accuracy') 
# plt.plot(epochs, accuracy_list) 
# plt.show()