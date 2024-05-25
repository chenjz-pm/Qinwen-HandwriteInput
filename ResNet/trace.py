import torch
from model import *
state_dict = torch.load('params.pth')

device = torch.device("cpu")
model=ResNet(Bottleneck,[3,4,6,3],10)
model.load_state_dict(torch.load("./params.pth",map_location=device))
model.eval()
model.to(device)

# 转换为 TorchScript 格式
script_model = torch.jit.trace(model, torch.randn(1, 3,96,96))

# 保存 TorchScript 模型为.pt文件
script_model.save('custom_model.pt')