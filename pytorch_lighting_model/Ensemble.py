import os
import torch
from torch import nn
import pytorch_lightning as pl
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import T5Model, T5Tokenizer
import re
import math
from pytorch_lighting_model.ScoreAttention import MultiInputAttention
from pytorch_lighting_model.ScoreAttention import GatedMultiInputAttention
from pytorch_lighting_model.ScoreAttention import MultiInputAttention2
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
        # 这个在注意力机制文件中
        # 第一次为128
        # 第二次为256
        # 第三次为64

        self.attention = MultiInputAttention(channel=model_dim, hidden_size=512, attention_size=128)
        # 测试一下下面的模型
        # self.attention = GatedMultiInputAttention(channel=model_dim, hidden_size=512, attention_size=128)

        #同时修改了下面要记住
        # self.attention = MultiInputAttention2(channel=model_dim, hidden_size=512, attention_size=128)

        self.fc = nn.Sequential(
            nn.Linear(model_dim*3 , 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x, nano_data):
        print("x",x)
        print("nano_data",nano_data)
        # 先把三个模型的梯度给停掉
        '''

        '''
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
        # 传递给对应的分类器
        rnn_rep = self.rnn_cls(rnn_rep)
        cnn_rep = self.cnn_cls(cnn_rep)
        bert_rep = self.bert_cls(bert_rep)

        # 三个模型抛出对应的数据以后,进行一个简单的分类
        repre = self.attention(rnn_rep, cnn_rep, bert_rep)

        if self.flag == 0:
            torch.save(rnn_rep, 'rnn.pt')
            torch.save(cnn_rep, 'cnn.pt')
            torch.save(bert_rep, 'bert.pt')
            torch.save(repre, 'edntom.pt')
            print("存储数据成功")
            self.flag = 1

        """
                
        """

        # 然后再用这个fc层,把三个输出最终变成一个
        out = self.fc(repre)



        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # x, nano_data, y = batch

        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = batch

        # calculate the accuracy
        output = self.forward(x, nano_data)

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
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_ACC", acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_AUPRC", AUPRC, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_AUROC", AUROC, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_Precision", Precision, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_Recall", Recall, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_F1Score", F1Score, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from pytorch_lightning.callbacks import LearningRateMonitor
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) #1e-3
        # return optimizer

        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=self.weight_decay) #1e-3
        # return optimizer

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # , weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        monitor = "val_AUPRC"  # 监视指标为验证集损失
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': monitor,
            'callbacks': [lr_monitor]
        }

    def validation_step(self, val_batch, batch_idx):
        # self.eval()
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = val_batch



        # calculate the accuracy
        output = self.forward(x, nano_data)

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

        if self.flag == 1:
            torch.save(y, 'label.pt')
            print("标签存储成功")
            self.flag = 2
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
        # AUPRC = torch.tensor(AUPRC, dtype=float).cuda()
        # AUROC = torch.tensor(AUROC, dtype=float).cuda()
        # Precision = torch.tensor(Precision, dtype=float).cuda()
        # Recall = torch.tensor(Recall, dtype=float).cuda()
        # F1Score = torch.tensor(F1Score, dtype=float).cuda()
        # 返回损失值字典
        # print('val_loss:', loss)
        return {'val_loss': loss, 'val_ACC': acc, 'val_AUPRC': AUPRC, 'val_AUROC': AUROC, 'val_Precision': Precision,
                'val_Recall': Recall, 'val_F1Score': F1Score, }

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
        # print('avg_val_loss:', avg_loss)

    # def on_validation_end(self):
    #     # 强制将验证集损失值写入日志系统
    #     self.log('val_loss', self.trainer.callback_metrics['val_loss'], on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = test_batch

        # calculate the accuracy
        output = self.forward(x, nano_data)

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

        if self.flag==1:
            torch.save(y,'label.pt')
            self.flag=2

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
        self.log("test_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
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
