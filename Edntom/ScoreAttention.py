# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 21:45
# @Author  : Yin Chenglin
# @Email   : sdyinruichao@163.com
# @IDE     : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInputAttention(nn.Module):
    # 256 512
    # 原本为128
    # 调整第一次为256
    def __init__(self, channel=256, hidden_size=512, attention_size=256):
        super(MultiInputAttention, self).__init__()
        self.channel = channel           # 特征的通道数目
        self.hidden_size = hidden_size   # 隐藏数目
        self.attention_size = attention_size # 注意力的大小
        self.flag=0

        # 将输入的特征矩阵转换到统一的隐藏空间大小
        self.feature_transform = nn.Linear(channel, hidden_size)

        # 注意力网络
        self.attention_network = nn.Sequential(
            nn.Linear(hidden_size*3, attention_size),
            nn.Tanh(),#nn.ReLU(),#
            nn.Linear(attention_size, 3)
        )

    def forward(self, x1, x2, x3):  # x1 x2 x3应该是三个模型的预测结果 torch.Size([100, 256])
        # 将输入矩阵转换为(batch, length, hidden_size),三个输入的张量统一变成torch.Size([100, 512])

        # 先mask后面的一半
        # x1[:, :] = 0
        # x2[:, :] = 0
        # x3[:, :] = 0

        x1_transformed = self.feature_transform(x1)
        x2_transformed = self.feature_transform(x2)
        x3_transformed = self.feature_transform(x3)

        # 合并所有特征     torch.Size([100, 1536])
        combined_features = torch.cat((x1_transformed, x2_transformed, x3_transformed), dim=1)




        # 应用注意力机制   torch.Size([100, 3])
        attention_weights = self.attention_network(combined_features)

        attention_weights = F.softmax(attention_weights, dim=1)

        # 分割权重以对应三个输入
        split_attention_weights = attention_weights.split(1, dim=1)



        # 加权特征并拼接,每个张量都包含一个元素作为权重,乘以对应的输入

        weighted_x1 = x1 * split_attention_weights[0] # (100,1),并且这100个元素的和为1
        weighted_x2 = x2 * split_attention_weights[1] # x为[100,256]
        weighted_x3 = x3 * split_attention_weights[2] # 广播机制实现了元素乘法,还是(100,256)

        # 把三个输入在做一个拼接         拼接以后就是(100,256*3)
        combined_weighted_features = torch.cat((weighted_x1, weighted_x2, weighted_x3), dim=1)
        # combined_weighted_features = torch.cat((x1,x2,x3), dim=1)




        return combined_weighted_features


