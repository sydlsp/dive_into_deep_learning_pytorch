import torch
x=torch.arange(4.0)
print (x)
"""下面要对y=2XTX 对X进行求导
   在求导之前为了避免出现每次更新参数的时候把内存都消耗完 要开辟一个区域存储梯度"""

x.requires_grad_(True)#实际上等于打开自动求导
print(x.grad)

y=2*torch.dot(x,x)
y.backward()
print(x.grad)