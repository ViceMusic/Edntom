import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torcheval.metrics import BinaryAUPRC, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall

nc_dic ={'A':0, 'T':1, 'G':2, 'C':3, 'N':4}
device='cuda:0'

# sequence classification model
# fixed baseline model as in the deepMOD
class biRNN_basic(pl.LightningModule):
    def __init__(self, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN_basic, self).__init__()
        # self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)\
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()

    def forward(self, x, nano_data):
        seqs = list(x)
        seq_encode = [[nc_dic[c] for c in f] for f in seqs]
        seq_encode = torch.Tensor(seq_encode).to(torch.int64)
        input_ids = F.one_hot(seq_encode)
        input_ids = input_ids.cuda()
        # print(input_ids.size())
        # print(nano_data.size())
        x = torch.cat((input_ids, nano_data), -1)

        # 设置初始的隐藏状态,细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()

        # out, _ = self.lstm(self.embed(x), (h0, c0))
        rnn_out, _ = self.lstm(x, (h0, c0))

        # linear-1x, logits output
        out = self.fc(rnn_out[:, int(x.size(1) / 2)+1, :])

        return out, rnn_out[:, int(x.size(1) / 2)+1]
