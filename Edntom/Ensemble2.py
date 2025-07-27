import os
import torch
from torch import nn
import pytorch_lightning as pl
from Edntom.ScoreAttention import MultiInputAttention
import torch.nn.functional as F
from torcheval.metrics import BinaryAUPRC, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall

# 注: 这个模型只是为了消融实验设计的, 闲着没事不要跑这个!
class Ensemble(pl.LightningModule):
    def __init__(self, RNN, BERT, CNN, model_dim=256):
        super().__init__()
        # 将三个模型都作为集成的输出
        # 在默认的情况下这三个模型是收到影响的
        self.rnn_model = RNN
        self.cnn_model = CNN
        self.bert_model = BERT
        self.flag=0

        # 将输出从700--->256整个范围
        self.bert_cls = nn.Sequential(
            nn.BatchNorm1d(700),
            nn.ReLU(),
            nn.Linear(700, model_dim),
        )

        self.rnn_cls = nn.Sequential(
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, model_dim),
        )

        self.cnn_cls = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, model_dim),
        )

        self.attention = MultiInputAttention(channel=model_dim, hidden_size=512, attention_size=128)

        self.fc = nn.Sequential(
            nn.Linear(model_dim*3 , 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        # 非常粗暴的计算方式
        self.cls = nn.Linear(model_dim,2)

    def forward(self, x, nano_data):
        for param in self.rnn_model.parameters():
            param.requires_grad = False

        for param in self.cnn_model.parameters():
            param.requires_grad = False

        for param in self.bert_model.parameters():
            param.requires_grad = False

        # 进行计算
        _, rnn_rep = self.rnn_model(x, nano_data)
        cnn_rep, _ = self.cnn_model(x, nano_data)
        _, bert_rep = self.bert_model(x, nano_data)
        # 传递给对应的分类器, 得到的长度均为[1000,256]
        rnn_rep = self.rnn_cls(rnn_rep)
        cnn_rep = self.cnn_cls(cnn_rep)
        bert_rep = self.bert_cls(bert_rep)
        # 直接变成[1000,2]  [未甲基化概率, 甲基化概率]
        rnn_rep=torch.softmax(self.cls(rnn_rep), dim=1)
        cnn_rep=torch.softmax(self.cls(cnn_rep), dim=1)
        bert_rep=torch.softmax(self.cls(bert_rep), dim=1)

        # 获取每一个模型对于"是否甲基化"的判断
        rnn_prob = rnn_rep[:, 1].requires_grad_(True)  # [1000]
        cnn_prob = cnn_rep[:, 1].requires_grad_(True)
        bert_prob = bert_rep[:, 1].requires_grad_(True)

        # 使用显式加法避免原地操作, 直接原地相加
        total_probs = rnn_prob.add(cnn_prob).add(bert_prob)  # 替代连续+操作

        # 温度参数需保持可导性, 设置阈值为1.5
        T = torch.tensor(0.1, requires_grad=True)  # 网页8
        mask = torch.sigmoid((total_probs - 1.5) / T)

        # 构建梯度桥接（网页3）
        final_labels = torch.stack([1 - mask, mask], dim=1).clone()  # 克隆新张量

        return final_labels


