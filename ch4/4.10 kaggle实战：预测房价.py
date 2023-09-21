import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

train_data=pd.read_csv("D:\\动手学深度学习数据集\\train.csv")
test_data=pd.read_csv("D:\\动手学深度学习数据集\\test.csv")

"""
对数据集的行列数有个简要了解
"""
print(train_data.shape)
print(test_data.shape)

"""
通过观察数据集特征，这个可以直接通过打开csv文件看出来，数据的第一个标签是ID，显然这不能作为特征，所以要把这个给去除掉
train_data的最后一列是价格而test_data最后一列不是，也不能把价格作为特征，因此按照如下方式处理数据集
"""

#.iloc是按照行列索引来获取数据 还有.loc方式它是按照行列标签来进行索引
#pd.concat是连接函数，默认是堆起来连接
all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

"""
对数值的处理
"""
# print(all_features.shape)

#获取每一列的属性
#all_features.dtypes
# 语句结果：
# MSSubClass         int64
# MSZoning          object
# LotFrontage      float64
# LotArea            int64
# Street            object
#                   ...
# MiscVal            int64
# MoSold             int64
# YrSold             int64
# SaleType          object
# SaleCondition     object
# Length: 79, dtype: object

# print(type(all_features.dtypes !='object')

#取出数字类型的标签名
numeric_features=all_features.dtypes[all_features.dtypes !='object'].index  #这里的语法我暂时理解有点像根据标签查数据
#上面这句话的具体理解看上方的注释以及https://blog.csdn.net/weixin_52024290/article/details/132129907文章内容
print(numeric_features)
#这里.apply函数默认对列进行操作 lambda语法实际上是一个无名函数，理解为函数操作就可以了，这句话的意思是把数值部分的列转化为均值为0方差为1的数据
#all_features[numeric_features]这里同样可以理解为根据标签取数据，一系列操作有点像数据库取数  (目前理解)
all_features[numeric_features]=all_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))#想改成行操作写成apply(lambda x:(x-x.mean())/(x.std()),axis=1)就好了\
#这句话为了解决在标准化过程中出现的nan的问题，把nan变成0，也就是均值
all_features[numeric_features]=all_features[numeric_features].fillna(0)

"""
对离散值的处理，说白了就是生成独热编码
"""
all_features=pd.get_dummies(all_features,dummy_na=True) #pd.get_dummies是pandas生成独热编码的方式 dummy_na表示关注NAN

# print(all_features.shape)

"""
从pandas格式提取Numpy格式，并将其转化为张量表示
"""
n_train=train_data.shape[0]
train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float32)  #.values表示把all_features[:n_train]里面的二维数组取出来
test_features=torch.tensor(all_features[n_train:].values,dtype=torch.float32)
train_labels=torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)  #表示把SalePrice的值给取出来并规定一下形状


"""
训练
"""

loss=nn.MSELoss()#采用均方误差作为损失函数

in_features=train_features.shape[1]

def get_net():
    net=nn.Sequential(nn.Linear(in_features,1))
    return net

#在房价预测中，考虑到房价的基准水平可能会差的比较大，采用相对误差来进行衡量更为合适，一般来言相对的东西是用除法的，这里采用了log变除法为减法的技巧

def log_rmse(net,features,labels):
    clipped_preds=torch.clamp(net(features),1,float('inf'))  #.clamp的作用是把输出结果每个数的范围限制到[1,无穷大)之间
    rmse=torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()  #.item操作是取张量内的元素值，并返回该元素，保持元素值不变

def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls=[],[]
    #train_iter=d2l.load_array((train_features,train_labels),batch_size)
    #上面这句话我认为等价于
    dataset = data.TensorDataset(train_features,train_labels)  #实际上有点类似于一个zip过程
    train_iter=data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

"""
k折交叉验证
"""
def get_K_fold_data(k,i,X,y):
    assert k>1
    fold_size=X.shape[0]//k  #每一折里包含的样本数
    X_train,y_train=None,None
    for j in range(k):
         idx=slice(j*fold_size,(j+1)*fold_size)
         X_part,y_part=X[idx,:],y[idx]  #分出每一折数据元素
         if (j==i):  #如果要用第i折为验证集
             x_valid,y_valid=X_part,y_part
         elif X_train is None:
             X_train, y_train=X_part, y_part
         else:
             X_train=torch.cat([X_train,X_part],0)
             y_train=torch.cat([y_train,y_part],0)
    return X_train,y_train,x_valid,y_valid


"""
利用k折交叉验证网络
"""
def k_fold(k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum,valid_l_sum=0,0
    for i in range (k):
        data=get_K_fold_data(k,i,X_train,y_train)
        net=get_net()
        train_ls,valid_ls=train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

"""
实际运行
"""

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
#                           weight_decay, batch_size)
# print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
#       f'平均验证log rmse: {float(valid_l):f}')

"""
kaggle提交
"""
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size) #在训练集上训练数据
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集,detach的作用是将pred从计算图中分离下来
    preds = net(test_features).detach().numpy()  #得到测试数据
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])  #这里相当于操作数据库 在test_data表里面加一列SalePrice
    #test_data.to_csv("test.csv",index=False)  将修改保存到文件里
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)

"""
这次提交kaggle测评实际上是提交了生成的submission.csv文件
"""

"""
总结：
本次kaggle实战房价预测主要有如下几个过程组成：
1.读取csv文件(读取数据集)
2.对数据集处理：包括对数值型数据进行归一化，对离散型数据生成独热编码
3.定义网络，损失函数
4.定义训练过程，定义k折交叉验证，网络训练(k折交叉验证其实是对训练集来言，将其分为训练集和验证集，网上下载的test.csv其实是测试集)
5.在整个数据集上跑一遍，获得最终csv文件，提交

目前理解的pandas和Numpy发挥的作用：
pandas用于处理表的数据，在实际投放到网络的时候要转化为Numpy张量来做
"""