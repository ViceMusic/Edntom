# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 16:15
# @Author  : Yin Chenglin
# @Email   : sdyinruichao@163.com
# @IDE     : PyCharm

import torch
from GPUtil import GPUtil

# from util.data_classification import load_classfication_data, make_data, MyDataSet
from pytorch_lighting_model.BERT_plus_I import BERT_plus_encoder
from pytorch_lighting_model.RNN_I import biRNN_basic
from pytorch_lighting_model.CNN_rep import CNN
from pytorch_lighting_model.Ensemble import Ensemble
from utils import load_dataset1, make_data, MyDataSet
from torch.utils.data import DataLoader
# from pytorch_lighting_model.lighting_slice import model_encoder
# from model.classification_attention import MultiModalModel
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl


if __name__ == '__main__':

    #data_load = load_dataset1("/mnt/sde/tiange/Nanopore_data/humen1.csv") # 人类数据集
    data_load = load_dataset1("./data/balance_tha.csv")  # 拟南芥数据集
    #data_load = load_dataset1("./data/balance_rice.csv") # 水稻数据集

    sequences, nano, label = make_data(data_load)
    dataset = MyDataSet(sequences, nano, label)

    proportions = [.9, .1]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    mnist_train, mnist_val = random_split(dataset, lengths)

    torch.save(mnist_train, "./dataset/tha_train.pt")
    torch.save(mnist_val, "./dataset/tha_val.pt")
    print('拟南芥数据集划分完成')