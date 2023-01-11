import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from model import *
import time

#创建数据集
train_dataset = torchvision.datasets.CIFAR10('./dataset',train=True,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10('./dataset',train=False,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)

#数据集长度
train_dataset_len = len(train_dataset)
test_dataset_len = len(test_dataset)
print('训练集的长度为:{}'.format(train_dataset_len))
print('测试集的长度为:{}'.format(test_dataset_len))

#利用dataloader加载数据集
train_dataloader = DataLoader(train_dataset,batch_size=64)
test_dataloader = DataLoader(test_dataset,batch_size=64)

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(fen_cifar10.parameters(),lr=learning_rate)

#训练网络的参数
#训练次数
total_train_step = 0
#测试次数
total_test_step = 0
#训练轮数
epoch = 10

for i in range(epoch):
    print('-----第{}次训练开始-----'.format(i+1))
    start_time = time.time()
    #训练模型
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        output = fen_cifar10(imgs)
        loss = loss_fn(output,targets) #计算误差

        #优化器优化过程
        optimizer.zero_grad()   #梯度清零
        loss.backward()  #误差反向传播
        optimizer.step() #对反向传播的结果进行优化

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print('训练次数为:{},Loss:{}'.format(total_train_step,loss.item()))
            print('训练时长:{}s'.format(end_time-start_time))

    #测试模型
    total_test_loss = 0
    total_accuracy = 0
    # 阻止反向传播，即不调优
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = fen_cifar10(imgs)
            #计算误差值
            loss = loss_fn(output,targets)
            total_test_loss = total_test_loss + loss.item()
            #计算正确率
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
        total_test_step += 1
        print('整体测试集上的Loss:{}'.format(total_test_loss))
        print('整体测试集上的正确率:{}'.format(total_accuracy/test_dataset_len))