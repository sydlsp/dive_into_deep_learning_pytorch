import torch
"""两个样本在三个类别上的分类"""
y=torch.tensor([0,2])  #表示第一个样本 类别0是正确预测 第二个样本类别2是正确预测
y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# print (y_hat[0][y[0]])
# print (y_hat[1][y[1]])

def cross_entropy(y_hat,y):
    """"""
    """书上给出的代码  书上给的代码是这个意思 y_hat[0][0]和y_hat[0,0]表示的意思是一样的 看懂了这个下面就好理解了"""
    print (y_hat[range(len(y_hat)),y])
    return -torch.log(y_hat[range(len(y_hat)),y])
    """自己写的代码 
       所谓的交叉熵就是 ∑yilogyi_hat 
       对于第一个样本来言 真实的值就是[1,0,0] 对于第二个样本是[0,0,1]  书本给出的代码实际上是忽略了那个1"""
    return -1*torch.log(y_hat[0][y[0]]),-1*torch.log(y_hat[1][y[1]])

print (cross_entropy(y_hat,y))
print(y_hat[0,2])