import torch
from torch import nn

#需要分类的类别数
classes=5

#SE模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class Res2NetBottleneck(nn.Module):
    expansion = 4  #残差块的输出通道数=输入通道数*expansion
    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=True,  norm_layer=True):
        #scales为残差块中使用分层的特征组数，groups表示其中3*3卷积层数量，SE模块和BN层
        super(Res2NetBottleneck, self).__init__()

        if planes % scales != 0: #输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        if norm_layer:  #BN层
            norm_layer = nn.BatchNorm2d

        bottleneck_planes = groups * planes
        self.scales = scales
        self.stride = stride
        self.downsample = downsample
        #1*1的卷积层,在第二个layer时缩小图片尺寸
        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, stride=stride)
        self.bn1 = norm_layer(bottleneck_planes)
        #3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv2 = nn.ModuleList([nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales,
                                              kernel_size=3, stride=1, padding=1, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        #1*1的卷积层，经过这个卷积层之后输出的通道数变成
        self.conv3 = nn.Conv2d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        #SE模块
        self.se = SEModule(planes * self.expansion) if se else None

    def forward(self, x):
        identity = x

        #1*1的卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #scales个(3x3)的残差分层架构
        xs = torch.chunk(out, self.scales, 1) #将x分割成scales块
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        #1*1的卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        #加入SE模块
        if self.se is not None:
            out = self.se(out)
        #下采样
        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class Res2Net(nn.Module):
    def __init__(self, layers, num_classes, width=16, scales=4, groups=1,
                 zero_init_residual=True, se=True, norm_layer=True):
        super(Res2Net, self).__init__()
        if norm_layer:  #BN层
            norm_layer = nn.BatchNorm2d
        #通道数分别为64,128,256,512
        planes = [int(width * scales * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]

        #7*7的卷积层，3*3的最大池化层
        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #四个残差块
        self.layer1 = self._make_layer(Res2NetBottleneck, planes[0], layers[0], stride=1, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer2 = self._make_layer(Res2NetBottleneck, planes[1], layers[1], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(Res2NetBottleneck, planes[2], layers[2], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer4 = self._make_layer(Res2NetBottleneck, planes[3], layers[3], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        #自适应平均池化，全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3] * Res2NetBottleneck.expansion, num_classes)

        #初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #零初始化每个剩余分支中的最后一个BN，以便剩余分支从零开始，并且每个剩余块的行为类似于一个恒等式
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Res2NetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, scales=4, groups=1, se=True, norm_layer=True):
        if norm_layer:
            norm_layer = nn.BatchNorm2d

        downsample = None  #下采样，可缩小图片尺寸
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample, stride=stride, scales=scales, groups=groups, se=se, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scales=scales, groups=groups, se=se, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = nn.functional.softmax(logits, dim=1)

        return probas
import torchvision

model=Res2Net([2,2,2,2],10,groups=32,width=4)
model.cuda()
optimizer=torch.optim.Adam(model.parameters(),lr=0.004)
lossfunc=nn.CrossEntropyLoss().cuda()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train = True, 
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset ,batch_size = 256, shuffle = True)


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('./logs')

epochs,allstep=2,0
from tqdm import tqdm
with tqdm(total=epochs,desc="epoch") as pbar:
    for epoch in range(epochs):
        for step, (data, targets) in enumerate(train_loader):
            data=data.cuda()
            targets=targets.cuda()
            output = model(data)

            optimizer.zero_grad()
            loss = lossfunc(output, targets)
            loss.backward()
            optimizer.step()

            allstep+=1
            print("step={},loss={}".format(allstep,loss.item()))
            # writer.add_scalar('loss', loss, allstep)
            # writer.add_scalar('epoch', epoch+1, step)
        #pbar.update(1)
    
torch.save(model.state_dict(), 'params.pth')