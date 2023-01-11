import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

#搭建神经网络-模型CIFAR10
class Fen(nn.Module):
    def __init__(self):
        super(Fen, self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,input):
        output = self.model1(input)
        return output
fen_cifar10 = Fen()
fen_cifar10 = fen_cifar10.cuda()

#调用神经网络-模型vgg16
fen_vgg16 = torchvision.models.vgg16()
fen_vgg16.classifier.add_module('7',Linear(in_features=1000,out_features=10))
fen_vgg16 = fen_vgg16.cuda()