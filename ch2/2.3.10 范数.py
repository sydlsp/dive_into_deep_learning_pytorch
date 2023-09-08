import torch
"""范数实际上是描述向量大小的量  torch.norm 计算的实际上是L2范数"""
u=torch.tensor([3.0,4.0])
length=torch.norm(u)
print (length)

"""同时在实际应用中还会用到L1范数 L1范数实际上是向量各分量绝对值的和"""

x=torch.tensor([-1,2,3],dtype=torch.float32)
print (torch.abs(x).sum())