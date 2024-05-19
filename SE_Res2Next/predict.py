from model import *

model=SE_Res2Net(Bottleneck,[3,2,2,2],10,groups=32,width_per_group=4)
model.cuda()
model.load_state_dict(torch.load("./params.pth"))

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = torchvision.datasets.CIFAR10(root='./data',train = False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size = 64,shuffle = False)

tr=0
for step, (data, targets) in enumerate(testloader):
    data=data.cuda()
    model.cuda()
    output = model(data[:1])
    pred=torch.argmax(output).item()
    real=targets[0].item()
    if pred == real:
        tr+=1 
    print(pred,'  ',real,'  ',tr)

