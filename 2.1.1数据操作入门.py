import torch
"""创建张量"""
x=torch.arange(12)
print (x)
"""查看向量的形状和元素总数"""
print(x.shape)#用于展示形状
print(x.numel())#用于计算元素个数
"""改变张量的形状"""
X=x.reshape(3,4)
#在reshape函数中并不需要计算出具体的行和列 比如要求三行reshape(3，-1)即可 系统会自动计算-1具体是多少
print(X)
print(X.shape)
print(X.numel())

"""创建数字初始化矩阵"""
y=torch.zeros((2,3,4))
#在这里要对2，3，4有理解 是创建2个三行四列的矩阵 2，3，4，3，4 实际上创建2*3*4个三行四列的矩阵
print (y)

z=torch.ones((2,3,4))
print(z)

a=torch.randn(3,4)
print (a)

b=torch.tensor([[1,2,3,4],[5,6,7,8]])
print (b)