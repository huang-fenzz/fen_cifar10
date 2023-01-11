import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

category = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
image_path = './imgs/yingying.jpg'
# image_path = './imgs/dog.png'
image = Image.open(image_path)
image = image.convert('RGB')  #png格式除了RGB外另有一个透明通道

#修改图片尺寸
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])
image = transform(image) #torch.Size([3, 32, 32])

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

#加载预训练参数
model = torch.load('./model/fen_cifar10_50.pth')

image = torch.reshape(image,(1,3,32,32))
image = image.cuda()
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print('图片类别属于：{}'.format(category[output.argmax(1).item()]))