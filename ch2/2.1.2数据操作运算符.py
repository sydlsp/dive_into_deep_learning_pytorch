import torch
x=torch.tensor([1,2,3,4.0])
y=torch.tensor([2,2,2.0,2])
"""按各元素运算"""
print (x+y)
print (x-y)
print (x*y)
print (x/y)
print (x**y)
z=torch.exp(x)
print (z)
"""张量的连接"""
a=torch.arange(12,dtype=torch.float32).reshape(3,4)
b=torch.tensor([[-1,-2,-3,-4],[-5,-6,-7,-8],[-9,-10,-11,-12]])
c=torch.cat((a,b),dim=0)#cat是连接运算函数，dim=0实际上就是变高了
print (c)
d=torch.cat((a,b),dim=1)#dim=1实际上是变长了
print(d)

"""逻辑运算符"""
print(a==b)

"""对张量中所有元素求和"""
print(a.sum())