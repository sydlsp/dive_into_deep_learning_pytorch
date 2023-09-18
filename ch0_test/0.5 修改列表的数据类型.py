import torch

y=torch.tensor([1,2])
y_hat=torch.tensor([3,4],dtype=torch.float32)
print(id(y))
print (y)
y=y.type(y_hat.dtype)  #相当于修改了y的数据类型
print (id(y))
print(y)