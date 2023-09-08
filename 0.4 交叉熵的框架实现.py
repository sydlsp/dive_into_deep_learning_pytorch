import torch
from torch import nn

predict=torch.tensor([0,0],dtype=torch.float32)
true=torch.tensor([0,1],dtype=torch.float32)
print (predict)
print(true)
loss=nn.CrossEntropyLoss()
print(loss(predict,true))


