import torch
X=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)
num_inputs=784
num_outputs=10
def Softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)  #按行求和
    return X_exp/partition  #这里应用到了广播机制



print (Softmax(X))
w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
print(w.shape[0])
