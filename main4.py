
from dataset import dataset
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

from cv2 import transform
from matplotlib import transforms
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

import cv2
import numpy as np

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

print('device 개수:', torch.cuda.device_count())
print('현재 device:', torch.cuda.current_device())

# bring data & normalize
stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(*stats)])

trainset = dataset.C100Dataset(train=True, transfrom=transform)
testset = dataset.C100Dataset(train=False, transfrom=transform)

# loss function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# accuracy
def accuracy(output, y):
    return torch.sum(torch.argmax(output, axis=1)==torch.argmax(y, axis=1))

# num_epochs
#n_epochs = 50
n_epochs = 3

# loss & acc
tr_loss, vl_loss, test_loss = np.zeros(n_epochs, dtype='float64'), np.zeros(n_epochs, dtype='float64'), np.zeros(n_epochs, dtype='float64')
tr_accy, vl_accy, ts_accy = np.zeros_like(tr_loss), np.zeros_like(tr_loss), np.zeros_like(tr_loss)

kfold = KFold(n_splits = 4, random_state = 0, shuffle = True)
for fold, (train_idx, val_idx) in enumerate(kfold.split(trainset)):
    print("FOLD", fold)
    print('==================================================')

    data_size = len(trainset)
    subtrain_size = int(0.8*data_size)
    val_size = data_size - subtrain_size
    subtrain_set, val_set =  random_split(trainset, [subtrain_size, val_size])

    # data loader
    subtrain_loader = DataLoader(subtrain_set, batch_size=128, shuffle = True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=100, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, drop_last=False)

    # bring model
    print("===> Bring Model")
    model = resnet34()
    if torch.cuda.is_available():
        model = model.cuda()

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # running sub_training loop for num epochs
    print('==> training')
    
    for epoch in range(n_epochs):
        print('[Epoch : %d ] ' %epoch)
        train_loss = 0
        correct = 0
        total = 0
        
        model.train()
        for batch_idx, (img, target) in enumerate(subtrain_loader):
            img, target = img.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
            if batch_idx % 100 == 0 :
                print(batch_idx, len(subtrain_loader), 'Loss : %.3f | ACC : %.3f %% (%d/%d)'
            %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('training process finished')

    # validation
    print('validate & saving model')

    # saving model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)

    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        for batch_idx, (img, target) in enumerate(val_loader):
            img, target = img.cuda(), target.cuda()
            output = model(img)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct = predicted.eq(target).sum().item()
    
            if batch_idx%100 == 0:
                print(batch_idx, len(val_loader), 'Loss : %.3f | Acc : %.3f %% (%d/%d)'
                        %(test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

        # tr_loss[epoch] = train_loss
        # tr_accy[epoch] = accyt
        # vl_loss[epoch] = lossv
        # vl_accy[epoch] = accyv

        # print("Epoch:", epoch, "Train Loss:", train_loss, "Validation Loss:", vl_loss[epoch], \
        #     "Train Accuracy:", tr_accy[epoch]/(cnt-1), "Validation Accuracy:",vl_accy[epoch]/(cnt2-1))


