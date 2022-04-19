
from dataset import dataset
from model_resnet import resnet34

import matplotlib.image as img

import torch
from torch import batch_norm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as img

from cv2 import transform
from matplotlib import transforms
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

print('device 개수:', torch.cuda.device_count())
print('현재 device:', torch.cuda.current_device())

# bring data & normalize
stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
transform = transforms.Compose([
                transforms.RandomCrop(32, padding =4),
                transforms.RandomHorizontalFlip(),
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(*stats)])

trainset = dataset.C100Dataset(train=True, transfrom=transform)
testset = dataset.C100Dataset(train=False, transfrom=transform)

# num_epochs
n_epochs = 25

# data split
generator = torch.Generator()
generator.manual_seed(0)
data_size = len(trainset)
subtrain_size = int(0.8*data_size)
val_size = data_size - subtrain_size
subtrain_set, val_set =  random_split(trainset, [subtrain_size, val_size], generator=generator)

# data loader
subtrain_loader = DataLoader(subtrain_set, batch_size=128, shuffle = True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=100, shuffle=True, num_workers=2)
full_trainloader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, drop_last=False)

print("===> Bring Model")
model = resnet34()

if torch.cuda.is_available():
    model = model.cuda()

# loss & optimizer
criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay= 1e-5)

print('==> training')


subtrain_do = False

if subtrain_do:
    loss_list_subtrain = []
    accuracy_list_subtrain = []
    loss_list_val = []
    accuracy_list_val =[]
    # subtrain & val
    for epoch in range(n_epochs):
        print('[Epoch : %d] ' %epoch)
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
            
            if batch_idx % 50 == 0 :
                print(batch_idx, len(subtrain_loader), 'Loss : %.3f | ACC : %.3f %% (%d/%d)'
            %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        accuracy_list_subtrain.append(100. * correct / total)
        print('\nTotal train accuarcy:', 100. * correct / total)
        loss_list_subtrain.append(train_loss)
        print('Total train loss:', train_loss)

        print('training process finished')

        # validation
        print('validation')

        with torch.no_grad():
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            for batch_idx, (img, target) in enumerate(val_loader):
                img, target = img.cuda(), target.cuda()
                output2 = model(img)
                loss = criterion(output2, target)

                val_loss += loss.item()
                _, predicted = output2.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                if batch_idx%100 == 0:
                    print(batch_idx, len(val_loader), 'Loss : %.3f | Acc : %.3f %% (%d/%d)'
                        %(val_loss/(batch_idx+1), 100.*correct/total, correct, total))

            accuracy_list_val.append(100. * correct / total)
            print('\nTest accuarcy:', 100. * correct / total)
            loss_list_val.append(val_loss)
            print('Test average loss:', val_loss / total)


accuracy_list_train = []
accuracy_list_test =[]

# training full train dataset
for epoch in range(n_epochs):
    print('[Epoch : %d] ' %epoch)
    f_train_loss = 0
    correct = 0
    total = 0

    model.train()
    # train
    for batch_idx, (img, target) in enumerate(full_trainloader):
        img, target = img.cuda(), target.cuda()
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        f_train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0 :
            print(batch_idx, len(subtrain_loader), 'Loss : %.3f | ACC : %.3f %% (%d/%d)'
        %(f_train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    accuracy_list_train.append(100. * correct / total)
    print('\nTotal train accuarcy:', 100. * correct / total)

    # test dataset
    with torch.no_grad():
        print('test')
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        for batch_idx, (img, target) in enumerate(test_loader):
            img, target = img.cuda(), target.cuda()
            outputs2 = model(img)
            loss = criterion(outputs2, target)

            test_loss += loss.item()
            _, predicted = outputs2.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx%100 == 0:
                print(batch_idx, len(test_loader), 'Loss : %.3f | Acc : %.3f %% (%d/%d)'
                    %(test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        accuracy_list_test.append(100. * correct / total)
        print('\nTest accuarcy:', 100. * correct / total)


epochs = np.arange(0,n_epochs)
plt.figure(figsize=(10,5))
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend(['train','test'])
plt.plot(epochs, accuracy_list_train) 
plt.plot(epochs, accuracy_list_test) 
plt.savefig('graph4.png')

print(accuracy_list_test)
print(max(accuracy_list_test))