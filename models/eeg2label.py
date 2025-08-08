'''
define the models here

models:
    mlpnet: basic net
    glfnet: used to get the global and local features of EEG signals
'''
import torch.nn as nn
import torch

class mlpnet(nn.Module):
    '''used to define basic net
    todo: specify the usage, what is the meaning of input_dim and out_dim
    args: 
        input_dim: 65*5 EEG features
        out_dim: 50, used to classify the video into 50 classes
    '''
    def __init__(self, input_dim, out_dim):
        super(mlpnet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )
        
    def forward(self, x):
        # x, which size is [batch_size, C, 5]
        out = self.net(x)
        return out




class glfnet(nn.Module):
    '''glfnet used to get the global and local features of EEG signals
    todo: specify the usage, why we bother to use two nets to get global and local features
    args: 
        input_dim and output_dim are same as mlpnet
        emb_dim: the dimension of the embedding features between global features, local features and the final output
    '''
    def __init__(self, input_dim, emb_dim, out_dim):
        super(glfnet, self).__init__()
        self.globalnet = mlpnet(input_dim, emb_dim)
        self.occipital_index = list(range(50*5, 62*5))
        self.occipital_localnet = mlpnet(len(self.occipital_index), emb_dim)
        self.linearnet = nn.Linear(emb_dim*2, out_dim)

    def forward(self, x):
        global_feature = self.globalnet(x)
        # occipital_x is the occipital part of the EEG features
        # 修复索引问题：x的形状应该是[batch_size, features]，所以使用维度1进行索引
        occipital_x = x[:, self.occipital_index]
        occipital_feature = self.occipital_localnet(occipital_x)
        out = self.linearnet(torch.cat((global_feature, occipital_feature), dim=1))
        return out