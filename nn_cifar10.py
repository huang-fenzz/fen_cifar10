import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Fen(nn.Module):
    def __init__(self):
        super(Fen, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,input):
        output = self.model1(input)
        return output
fen = Fen()
print(fen)
input = torch.ones((64,3,32,32))
writer = SummaryWriter('logs_cifar10')
writer.add_graph(fen,input)
writer.close()