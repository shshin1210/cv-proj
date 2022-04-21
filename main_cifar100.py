"""Train CIFAR100 with PyTorch."""
from __future__ import print_function

import argparse
import os

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms

from models import PreActResNet18
from utils import progress_bar
from torch.autograd import Variable
from dataset import datasets
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

print('device number :', torch.cuda.device_count())
print('current device:', torch.cuda.current_device())

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--exp', default='cifar100_mixup', type=str,
                    help='name of the experiment')
parser.add_argument('--mixup', action='store_true',
                    help='whether to use mixup or not')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

num_classes = 100

# bring data & normalize
stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
transform = transforms.Compose([
                transforms.RandomCrop(32, padding =4),
                transforms.RandomHorizontalFlip(),
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(*stats)])

trainset = datasets.C100Dataset(train=True, transfrom=transform)
testset = datasets.C100Dataset(train=False, transfrom=transform)

# data loader

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=True, num_workers=2, drop_last=False)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_{}/ckpt.t7'.format(args.exp))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = PreActResNet18(num_classes=num_classes)

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)

# Training


def shuffle_minibatch(inputs, targets, mixup=True):
    batch_size = inputs.shape[0] # input dim
    print(batch_size)

    rp1 = torch.randperm(batch_size)
    # rp1 is a random permutation of intergers from 1 to batch_size without repeating elements

    inputs1 = inputs[rp1]

    targets1 = targets[rp1] # torch.size([128])
    targets1_1 = targets1.unsqueeze(1) # torch.size([128,1])

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2] 
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1) # into (row,1)


    y_onehot = torch.FloatTensor(batch_size, num_classes)
    y_onehot.zero_() 
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1) # .scatter_(dim, index, src)

    y_onehot2 = torch.FloatTensor(batch_size, num_classes)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = numpy.random.beta(1, 1, [batch_size, 1]) # beta distribution 
    else:
        a = numpy.ones((batch_size, 1))
                                                       # a[..., None, None] = (128,1) -> (128,1,1,1) to 4 dim 
    b = numpy.tile(a[..., None, None], [1, 3, 32, 32]) # numpy.tile(A,reps) Construct an array by repeating A the number of times given by reps.

    inputs1 = inputs1 * torch.from_numpy(b).float() 
    inputs2 = inputs2 * torch.from_numpy(1 - b).float() 

    c = numpy.tile(a, [1, num_classes])

    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float() # multiply weights
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle

accuracy_list_train = []
accuracy_list_test =[]

def train(epoch):
    print('\n[Epoch : %d]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs_shuffle, targets_shuffle = shuffle_minibatch(
            inputs, targets, args.mixup)

        if use_cuda:
            inputs_shuffle, targets_shuffle = inputs_shuffle.cuda(), \
                targets_shuffle.cuda()

        optimizer.zero_grad()

        inputs_shuffle, targets_shuffle = Variable(
            inputs_shuffle), Variable(targets_shuffle)

        outputs = net(inputs_shuffle)
        m = nn.LogSoftmax()

        loss = -m(outputs) * targets_shuffle
        loss = torch.sum(loss) / 128
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        _, targets = torch.max(targets_shuffle.data, 1)
        correct += predicted.eq(targets).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Epoch %d, Training Loss: %.3f | Acc: %.3f%% (%d/%d)'  # noqa
                     % (epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))  # noqa
        
    accuracy_list_train.append(100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Epoch %d, Test Loss: %.3f | Acc: %.3f%% (%d/%d)'  # noqa
                     % (epoch, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))  # noqa

    accuracy_list_test.append(100. * correct / total)


scheduler = lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150], gamma=0.1)

n_epochs = 250
for epoch in range(start_epoch, start_epoch + n_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()

epochs = np.arange(0,start_epoch + n_epochs)
plt.figure(figsize=(10,5))
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend(['train','test'])
plt.plot(epochs, accuracy_list_train) 
plt.plot(epochs, accuracy_list_test) 
plt.savefig('cifar100-.png')

print(accuracy_list_test)
print(max(accuracy_list_test))