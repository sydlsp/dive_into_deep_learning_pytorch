import torch

list_A=torch.tensor([1,2,6])
list_B=torch.tensor([1,3,4])

print((list_A>list_B).float())
print((list_A>1).float())

print(list_A*list_B)