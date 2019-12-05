import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import torch.nn.functional as F

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


# 读取训练集的图片数据，并整理为规定的格式
class Codeimg(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = os.listdir(root)
        self.transforms = transform
    def one_hot(self, label):
        bp = torch.Tensor([])
        for i in range(len(label)):
            num = ord(label[i]) - 48
            if num > 9:
                num -= 7
                if num > 35:
                    num -= 6
            a = num
            b = torch.zeros(1, 62)
            b[:, a] = 1
            bp = torch.cat((bp, b), dim=1)
        return bp
    def __getitem__(self, index):
        image_path = self.paths[index]
        label = list(image_path)[:-4]
        label = self.one_hot(label).reshape(310)
        pil_image = Image.open(self.root + image_path)
        if self.transforms:
            data = self.transforms(pil_image)
        else:
            image_array = np.asarray(pil_image)
            data = torch.from_numpy(image_array)
        return data, label
    def __len__(self):
        return len(self.paths)

# 卷积层（用于提取图片中的特征信息）
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    # 前向传播
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# vggNet 神经网络
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=310):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.fc = nn.Linear(4608, num_classes)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 实例化 vggNet 神经网络
def ResNet18():
    return ResNet(ResidualBlock)

# ——————————————— 开始训练并优化参数——————————————————
data = Codeimg('train/train/', transform)
dataloader = DataLoader(data, batch_size=32, shuffle=True, drop_last=False)
img, label = data[0]

cnn = ResNet18()
cnn.cuda()
loss_fn = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(cnn.parameters())

# 神经网络固定流程（前向传播，计算损失，反向传播，更新权值）
for i in range(7):
    for j,(img,labels) in enumerate(dataloader):
        img = img.cuda()
        labels = labels.cuda()
        out = cnn(img)
        loss = loss_fn(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j % 100 == 0:
            print('i=%d j=%d Loss: %.5f' %(i,j,loss.item()))

# 将最优的参数保存在‘can2.pt’ 文件中
torch.save(cnn.state_dict(),'can2.pt')
# 将输出的结果整理为自己规定的格式
def uncode(code):
    biao = list()
    for i in range(len(code)):
        if code[i]<10:
            biao.append(chr(code[i]+48))
        elif 10<=code[i]<36:
            biao.append(chr(code[i]+55))
        else:
            biao.append(chr(code[i]+61))
    return biao

# 读取测试集中的图片数据，进行预测
class UnCodeimg(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = os.listdir(root)
        self.transforms = transform
    def __getitem__(self, index):
        image_path = self.paths[index]
        label = image_path
        if label != '.DS_Store':
            label = list(label)[:-4]
            label = int(''.join(label))
            pil_image = Image.open(self.root + image_path)
            if self.transforms:
                data = self.transforms(pil_image)
            else:
                image_array = np.asarray(pil_image)
                data = torch.from_numpy(image_array)
            return data, label
    def __len__(self):
        return len(self.paths)

data = UnCodeimg('test/test/', transform)
cnn = ResNet18()

# 将训练出来的最优的参数信息读取出来
cnn.load_state_dict(torch.load('can2.pt', map_location='cpu'))
cnn.eval()
chu = dict()

for i in range(len(data)):
    if (i+1)%1000 == 0:
        print('i = ', i+1)
    imgs, labels = data[i]
    imgs = torch.Tensor(imgs).reshape(1,1,30,150)
    output = cnn(imgs)
    output = output.view(-1, 62)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    out = uncode(output)
    chu[labels] = out

num = list()
labels = list()
for i in range(len(chu)):
    num.append(i)
    labels.append(''.join(chu[i]))

# 保存预测类别标签至文件中
np.savetxt('anboyang.csv',labels, fmt='%s')

