import torch
x=torch.arange(12)
y=torch.randn(12)
print (x)
print (y)
"""利用id函数查看python准确地址"""
before=id(y)
y=x+y
after=id(y)
print(before)
print(after)
"""经过上面的实验可以看出把y=x+y赋值后为结果分配了新的内存
   在深度学习的过程中参数量很大，按照这种方式更新将及其消耗内存
   所以常常采用原地执行更新的策略"""


"""具体操作的方式就是使用切片来完成原地更新"""
a=torch.arange(10,dtype=torch.float32)
b=torch.randn(10)
print (a)
print(b)
print (a)
print (id(a))
a[:]=a+b
print (a)
print (id(a))

