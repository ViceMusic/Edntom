import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pytorch_lightning as pl
import re
import math
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

    #
    # def getout(self, x, r):
    #     out = self.fc(r)
    #     return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # with torch.autograd.detect_anomaly():
        criterion = nn.CrossEntropyLoss()#weight=torch.FloatTensor([1, 1])).to(device)
        x, nano_data, y = batch

        output, _ = self.forward(x, nano_data)

        # calculate the accuracy
        acc_sum = (output.argmax(dim=1) == y).float().sum().item()
        n = y.shape[0]
        acc = acc_sum / n

        loss = criterion(output, y)
        # loss.backward(retain_graph=True)

        predictions = F.softmax(output, dim=1)
        predictions = predictions[:, 1]
        predictions = predictions.cpu()
        y = y.cpu()
        # print(predictions)
        # print(y)
        # print("11111prediction: ", predictions.size())
        # print("11111y: ", y.size())


        metric1 = BinaryAUPRC()
        metric1.update(predictions, y)
        AUPRC = metric1.compute()

        metric2 = BinaryAUROC()
        metric2.update(predictions, y)
        AUROC = metric2.compute()

        metric3 = BinaryPrecision()
        metric3.update(predictions, y)
        Precision = metric3.compute()

        metric4 = BinaryRecall()
        metric4.update(predictions, y)
        Recall = metric4.compute()

        metric5 = BinaryF1Score()
        metric5.update(predictions, y)
        F1Score = metric5.compute()

        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) #1e-3
        return optimizer

    def validation_step(self, val_batch, batch_idx):
        # self.eval()
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = val_batch

        output, _ = self.forward(x, nano_data)

        # calculate the accuracy
        acc_sum = (output.argmax(dim=1) == y).float().sum().item()
        n = y.shape[0]
        acc = acc_sum / n

        loss = criterion(output, y)

        predictions = F.softmax(output, dim=1)
        predictions = predictions[:, 1]
        predictions = predictions.cpu()
        y = y.cpu()

        # print("11111prediction: ", predictions.size())
        # print("11111y: ", y.size())

        metric1 = BinaryAUPRC()
        metric1.update(predictions, y)
        AUPRC = metric1.compute()

        metric2 = BinaryAUROC()
        metric2.update(predictions, y)
        AUROC = metric2.compute()

        metric3 = BinaryPrecision()
        metric3.update(predictions, y)
        Precision = metric3.compute()

        metric4 = BinaryRecall()
        metric4.update(predictions, y)
        Recall = metric4.compute()

        metric5 = BinaryF1Score()
        metric5.update(predictions, y)
        F1Score = metric5.compute()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        acc = torch.tensor(acc, dtype=float).cuda()

        # 返回损失值字典
        # print('val_loss:', loss)
        return {'val_loss': loss, 'val_ACC': acc, 'val_AUPRC': AUPRC, 'val_AUROC': AUROC, 'val_Precision': Precision, 'val_Recall': Recall, 'val_F1Score': F1Score,}

    def validation_epoch_end(self, outputs):
        # 计算并输出平均损失
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        avg_ACC = torch.stack([x['val_ACC'] for x in outputs]).mean()
        self.log('avg_val_ACC', avg_ACC, on_step=False, on_epoch=True, prog_bar=True)
        avg_AUPRC = torch.stack([x['val_AUPRC'] for x in outputs]).mean()
        self.log('avg_val_AUPRC', avg_AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        avg_AUROC = torch.stack([x['val_AUROC'] for x in outputs]).mean()
        self.log('avg_val_AUROC', avg_AUROC, on_step=False, on_epoch=True, prog_bar=True)
        avg_Precision = torch.stack([x['val_Precision'] for x in outputs]).mean()
        self.log('avg_val_Precision', avg_Precision, on_step=False, on_epoch=True, prog_bar=True)
        avg_Recall = torch.stack([x['val_Recall'] for x in outputs]).mean()
        self.log('avg_val_Recall', avg_Recall, on_step=False, on_epoch=True, prog_bar=True)
        avg_F1Score = torch.stack([x['val_F1Score'] for x in outputs]).mean()
        self.log('avg_val_F1Score', avg_F1Score, on_step=False, on_epoch=True, prog_bar=True)

    # def on_validation_end(self):
    #     # 强制将验证集损失值写入日志系统
    #     self.log('val_loss', self.trainer.callback_metrics['val_loss'], on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = test_batch

        output, rep = self.forward(x, nano_data)

        # calculate the accuracy
        acc_sum = (output.argmax(dim=1) == y).float().sum().item()
        n = y.shape[0]
        acc = acc_sum / n

        loss = criterion(output, y)

        predictions = F.softmax(output, dim=1)
        predictions = predictions[:, 1]
        predictions = predictions.cpu()
        y = y.cpu()

        metric1 = BinaryAUPRC()
        metric1.update(predictions, y)
        AUPRC = metric1.compute()

        metric2 = BinaryAUROC()
        metric2.update(predictions, y)
        AUROC = metric2.compute()

        metric3 = BinaryPrecision()
        metric3.update(predictions, y)
        Precision = metric3.compute()

        metric4 = BinaryRecall()
        metric4.update(predictions, y)
        Recall = metric4.compute()

        metric5 = BinaryF1Score()
        metric5.update(predictions, y)
        F1Score = metric5.compute()

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        torch.save(rep, "RNN_test.pt")
        torch.save(y, "RNN_label.pt")
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # 计算并输出平均损失
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        print('avg_test_loss:', avg_loss)


# current one test the positional shift
class biRNN_test(nn.Module):
    def __init__(self, device, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN_test, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        # out, _ = self.lstm(self.embed(x), (h0, c0))
        rnn_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(rnn_out[:, int(x.size(1) / 2) + 1, :])

        return out


class biRNN(nn.Module):
    def __init__(self, device, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.fc0 = nn.Linear(hidden_size * 2, 32)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        rnn_out, _ = self.lstm(x, (h0, c0))

        out = F.relu(self.fc0(rnn_out[:, int(x.size(1) / 2), :]))
        out = self.fc(out)

        return out


# 2020/08/31
class biRNN_test_embed(nn.Module):
    def __init__(self, device, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN_test_embed, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.featEmbed = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        # out, _ = self.lstm(self.embed(x), (h0, c0))
        rnn_out, _ = self.lstm(self.featEmbed(x), (h0, c0))
        out = self.fc(rnn_out[:, int(x.size(1) / 2), :])

        return out


# add residual
# residual augmented implementation of ElMo
class biRNN_residual(pl.LightningModule):
    def __init__(self, device, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN_residual, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2 + input_size, hidden_size, 1, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(607, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # different layer implementation
        rep = [x[:, int(x.size(1) / 2), :]]
        # h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        # initial layer processing
        h, _ = self.lstm1(x)
        rep.append(h[:, int(x.size(1) / 2), :])

        for i in range(1, self.num_layers):
            ch = torch.cat([h, x], -1)
            h, _ = self.lstm2(ch)

            rep.append(h[:, int(x.size(1) / 2), :])

        # first dimention is the sample
        rep = torch.cat(rep, dim=-1)

        out = self.fc(rep)
        out = self.fc2(out)

        return out
