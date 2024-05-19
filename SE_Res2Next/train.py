from model import *

model=SE_Res2Net(Bottleneck,[3,2,2,2],10,groups=32,width_per_group=4)
model.cuda()
optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
lossfunc=nn.CrossEntropyLoss().cuda()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train = True, 
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset ,batch_size = 256, shuffle = True)


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('./logs')

epochs,allstep=1,0
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