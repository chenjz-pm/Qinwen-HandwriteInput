from model import *
import torchvision

model=ResNet(Bottleneck,[3,4,6,3],10)
#,groups=32,width_per_group=4
model.load_state_dict(torch.load("./params.pth"))
model.eval()
model.cuda()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Resize(96),
                                torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
testset = torchvision.datasets.CIFAR10(root='./data',train = False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size = 10,shuffle = True)

tr=0
for step, (data, targets) in enumerate(testloader):
    data=data.cuda()
    targets=targets.cuda()
    output = model(data[:1])
    pred=torch.torch.argmax(output)
    real=targets[0].item()
    if pred == real:
        tr+=1 
    print("{} pred={},tag={},tr={}".format(step+1,pred,real,tr/(step+1)))

