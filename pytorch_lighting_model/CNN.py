import os
import torch
from tensorflow.python.keras.activations import swish
from torch import nn
import pytorch_lightning as pl
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import T5Model, T5Tokenizer
import re
import math
import torch.nn.functional as F
from torcheval.metrics import BinaryAUPRC, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall

nc_dic ={'A':0, 'T':1, 'G':2, 'C':3, 'N':4}
device='cuda:0'

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CNN(pl.LightningModule):
    def __init__(self, size=128):
        super().__init__()

        self.sig_conv1 = nn.Conv1d(3, 4, 3)
        self.sig_bn1 = nn.BatchNorm1d(4)
        self.sig_conv2 = nn.Conv1d(4, 16, kernel_size=3, stride=1)  # Adjust kernel size and stride
        self.sig_bn2 = nn.BatchNorm1d(16)
        self.sig_conv3 = nn.Conv1d(16, size, kernel_size=3, stride=3)
        self.sig_bn3 = nn.BatchNorm1d(size)

        # Sequence pathway
        self.seq_conv1 = nn.Conv1d(4, 16, kernel_size=3, stride=1)  # Adjust kernel size and stride
        self.seq_bn1 = nn.BatchNorm1d(16)
        self.seq_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1)  # Adjust kernel size and stride
        self.seq_bn2 = nn.BatchNorm1d(32)
        self.seq_conv3 = nn.Conv1d(32, size, kernel_size=3, stride=3)
        self.seq_bn3 = nn.BatchNorm1d(size)

        # Merge pathway
        self.merge_conv1 = nn.Conv1d(size * 2, size, kernel_size=1, stride=1)  # Adjust kernel size and stride
        self.merge_bn1 = nn.BatchNorm1d(size)
        self.merge_conv2 = nn.Conv1d(size, size, kernel_size=1, stride=1)  # Adjust kernel size and stride
        self.merge_bn2 = nn.BatchNorm1d(size)
        self.merge_conv3 = nn.Conv1d(size, size, kernel_size=3, stride=2)
        self.merge_bn3 = nn.BatchNorm1d(size)
        self.merge_conv4 = nn.Conv1d(size, size, kernel_size=1, stride=2)
        self.merge_bn4 = nn.BatchNorm1d(size)

        self.fc = nn.Linear(size, 2)

        self.swish = Swish()

    def forward(self, x, nano_data):

        seqs = list(x)
        seq_encode = [[nc_dic[c] for c in f] for f in seqs]
        seq_encode = torch.Tensor(seq_encode).to(torch.int64)
        input_ids = F.one_hot(seq_encode)
        input_ids = input_ids.permute(0, 2, 1)
        seqs = input_ids.to(torch.float32).cuda()
        # input_ids = F.one_hot(seq_encode).to(torch.float32).cuda()
        nano_data = nano_data.permute(0, 2, 1)

        sigs_x = self.swish(self.sig_bn1(self.sig_conv1(nano_data)))
        sigs_x = self.swish(self.sig_bn2(self.sig_conv2(sigs_x)))
        sigs_x = self.swish(self.sig_bn3(self.sig_conv3(sigs_x)))

        seqs_x = self.swish(self.seq_bn1(self.seq_conv1(seqs)))
        seqs_x = self.swish(self.seq_bn2(self.seq_conv2(seqs_x)))
        seqs_x = self.swish(self.seq_bn3(self.seq_conv3(seqs_x)))
        # print(sigs_x.shape, seqs_x.shape)
        z = torch.cat((sigs_x, seqs_x), 1)

        z = self.swish(self.merge_bn1(self.merge_conv1(z)))
        z = self.swish(self.merge_bn2(self.merge_conv2(z)))
        z = self.swish(self.merge_bn3(self.merge_conv3(z)))
        z = self.swish(self.merge_bn4(self.merge_conv4(z)))

        z = torch.flatten(z, start_dim=1)
        z = self.fc(z)

        return z

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # with torch.autograd.detect_anomaly():
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = batch

        output= self.forward(x, nano_data)

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

        output = self.forward(x, nano_data)

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

        output = self.forward(x, nano_data)

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
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # 计算并输出平均损失
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        print('avg_test_loss:', avg_loss)

# data
# mnist_train, mnist_val = random_split(dataset, [55000, 5000])
#
# train_loader = DataLoader(mnist_train, batch_size=32)
# val_loader = DataLoader(mnist_val, batch_size=32)
#
# # model
#
# # training
# trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
# trainer.fit(model, train_loader, val_loader)
