import torch
"""我们说的点积 是两个向量的内积  利用torch.dot函数进行计算"""
x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,3,4,5],dtype=torch.float32)

print (torch.dot(x,y))

"""矩阵——向量积  利用torch.mv 实际上是matrix*vector"""
A=torch.arange(12,dtype=torch.float32).reshape(3,-1)
print (torch.mv(A,x))

"""矩阵——矩阵乘法 利用torch.mm   matrix*matrix  在使用的时候不要和达哈玛积混淆了"""
B=torch.arange(1,10,dtype=torch.float32).reshape(3,3)
C=torch.arange(1,10,dtype=torch.float32).reshape(3,3)
print (B)
print (C)
print (torch.mm(B,C))

