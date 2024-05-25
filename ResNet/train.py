from model import *
import torchvision
from torch.utils.tensorboard import SummaryWriter

#model=ResNet(Bottleneck,[3,4,6,3],10,groups=32,width_per_group=4)
model=ResNet(Bottleneck,[3,4,6,3],10)
model.cuda()
start_lr=0.002
optimizer=torch.optim.Adam(model.parameters(),lr=start_lr, weight_decay=1e-5)
lossfunc=nn.CrossEntropyLoss().cuda()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Resize(96),
                                torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
trainset = torchvision.datasets.CIFAR10(root='./data',train = True, 
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset ,batch_size = 150, shuffle = True)


# from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/tf_logs')

def setlr(nowlr):
    nowlr /= 4
    for param_group in optimizer.param_groups:
        param_group['lr'] = nowlr

epochs,allstep=8,0
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.67)
from tqdm import tqdm
with tqdm(total=epochs,desc="epoch") as pbar:
    for epoch in range(epochs):
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'],epoch)
        for step, (data, targets) in enumerate(train_loader):
            data=data.cuda()
            targets=targets.cuda()
            output = model(data)

            optimizer.zero_grad()
            loss = lossfunc(output, targets)
            nowloss = loss
            loss.backward()
            optimizer.step()
            allstep+=1
            writer.add_scalar('loss', loss, allstep)
        pbar.update(1)
        scheduler.step()
        torch.save(model.state_dict(), 'params.pth')