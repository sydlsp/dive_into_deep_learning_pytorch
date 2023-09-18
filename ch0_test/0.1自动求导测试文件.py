import torch
x=torch.tensor([2],dtype=torch.float32,requires_grad=True)
y=x**3
y.backward()
z=x**2
z.backward()
print (x.grad)
"""从以上代码我们可以得到y是x的函数 z也是x的函数 在用backward的时候相当于y对x求导，z也对x进行了求导
   要输出导数的结果用x.grad 但要注意的是这是结果是每次求导结果的累积  3x^2+2x=3*4+2*2=16"""

"""利用x.grad.zero_将导数累积结果清零"""
x.grad.zero_()
print (x.grad)
"""再次计算fx对x的导数就是正确的了"""
fx=5*x**2
fx.backward()
print(x.grad)
