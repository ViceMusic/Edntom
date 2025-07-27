import os
import torch
from torch import nn
import pytorch_lightning as pl
from Edntom.ScoreAttention import MultiInputAttention
import torch.nn.functional as F
from torcheval.metrics import BinaryAUPRC, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall


class Ensemble(pl.LightningModule):
    def __init__(self, RNN, BERT, CNN, model_dim=256):
        super().__init__()
        # 将三个模型都作为集成的输出
        # 在默认的情况下这三个模型是收到影响的
        self.rnn_model = RNN
        self.cnn_model = CNN
        self.bert_model = BERT
        self.flag=0
        # 下面是一些分类器层
        # 这几个分配器层
        '''
         self.bert_cls = nn.Sequential(
            nn.BatchNorm1d(700),
            nn.ReLU(),
            nn.Linear(700, 128),
        )
        '''
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

        torch.save(rnn_rep, 'rnn.pth')

        # 传递给对应的分类器, 得到的长度均为[1000,256]
        rnn_rep = self.rnn_cls(rnn_rep)
        cnn_rep = self.cnn_cls(cnn_rep)
        bert_rep = self.bert_cls(bert_rep)

        torch.save(rnn_rep, 'rnn.pth')
        torch.save(cnn_rep, 'cnn.pth')
        torch.save(bert_rep, 'bert.pth')



        # 三个模型抛出对应的数据以后,进行一个简单的分类
        repre = self.attention(rnn_rep, cnn_rep, bert_rep)

        torch.save(repre, 'out.pth')



        # 临时读取一波数据



        # 然后再用这个fc层,把三个输出最终变成一个
        out = self.fc(repre)


        exit("读取完成临时退出")



        return out


