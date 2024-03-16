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
    # data_load = load_pretrain_data("/mnt/sde/ycl/NanoCon/code/pretrain/nanopore_data/test.csv")
    # data_load_val = load_dataset("./dataset.pt")
    # enc_inputs_valid, contact_map_valid = make_data(data_load_val)
    # sequences, signal, stats = make_data(data_load)
    # dataset = MultiModalDataset("/mnt/solid/ycl/pretrain")#sequences, signal, stats)
    # dataset = MultiModalDataset("/mnt/sde/data/pretrain")

    # data_load = load_dataset1("/mnt/sde/ycl/NanoCon/code/Nanopore_data/rep_rice.csv")
    data_load = load_dataset1("/mnt/sde/tiange/Nanopore_data/humen1.csv") # 得到的是一个二维list

    # data_load = load_dataset1("./data/balance_rice.csv") # 得到的是一个二维list

    # [mnist_train, mnist_val] = torch.load("./Example_dataset.pt")
    sequences, nano, label = make_data(data_load)
    dataset = MyDataSet(sequences, nano, label)

    proportions = [.9, .1]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    mnist_train, mnist_val = random_split(dataset, lengths)

    

    # proportions = [.8, .1, .1]
    # lengths = [int(p * len(dataset)) for p in proportions]
    # lengths[-1] = len(dataset) - sum(lengths[:-1])
    # mnist_train, mnist_val, mnist_test = random_split(dataset, lengths)
    # data_load_t = load_dataset1("/mnt/sde/ycl/NanoCon/code/Nanopore_data/humen_test.csv")
    # sequence1, nano_data1, label1 = make_data(data_load_t)
    # mnist_val = MyDataSet(sequence1, nano_data1, label1)
    # torch.save([mnist_train, mnist_val], "./Example_dataset.pt")

    # # add test data
    # data_load_test = load_dataset1("/mnt/sdb/home/wrh/Nanopore_program/Nanopore_data/testing_set.csv")
    # sequence, nano_data, label = make_data(data_load_test)
    # test_dataset = MyDataSet(sequence, nano_data, label)

    # [mnist_train, mnist_val] = torch.load("./Example_dataset.pt")
    checkpoint_callback = ModelCheckpoint(
        # dirpath="./",
        monitor="avg_val_AUROC",
        filename="ENSEM-mnist-{epoch:02d}-{avg_val_loss:.5f}-{avg_val_ACC:.5f}-{avg_val_AUPRC:.5f}-{avg_val_AUROC:.5f}-{avg_val_Precision:.5f}-{avg_val_Recall:.5f}-{avg_val_F1Score:.5f}",
        save_top_k=1,
        mode="max",
        save_last=False
    )
    # loader = DataLoader(mnist_train, 1000, True)#, collate_fn=collate)#, num_workers=40)#, collate_fn=custom_collate_fn)#, num_workers=12)
    loader_valid = DataLoader(mnist_val, 1000, False)#, collate_fn=collate)#, num_workers=12) #, num_workers=32)#, collate_fn=custom_collate_fn)#,num_workers=12)
    #loader_test = DataLoader(mnist_test, 2000, False) #232

    #在测试数据集上选取2000数据
    #数据来源为水稻数据集
    #loader_test = DataLoader(mnist_val, 2000, False)

    RNN_path = '/mnt/sde/ycl/NanoCon/code/lightning_logs/version_163/checkpoints/RNN-mnist-epoch=49-avg_val_ACC=0.86791-avg_val_AUPRC=0.95966-avg_val_AUROC=0.93144-avg_val_Precision=0.89477-avg_val_Recall=0.90262-avg_val_F1Score=0.89864.ckpt'
    RNN = biRNN_basic()
    checkpoint = torch.load(RNN_path)
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}
    RNN.load_state_dict(checkpoint['state_dict'], strict=False)

    BERT_path = '/mnt/sde/ycl/NanoCon/code/lightning_logs/version_166/checkpoints/BERT-mnist-epoch=47-avg_val_ACC=0.88783-avg_val_AUPRC=0.96826-avg_val_AUROC=0.94672-avg_val_Precision=0.92553-avg_val_Recall=0.89940-avg_val_F1Score=0.91225.ckpt'
    BERT = BERT_plus_encoder(vocab_size=7, hidden=100, n_layers=3, attn_heads=4, dropout=0).float()
    checkpoint = torch.load(BERT_path)
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'linear' not in k}
    BERT.load_state_dict(checkpoint['state_dict'], strict=False)

    CNN_path = '/mnt/sde/ycl/NanoCon/code/lightning_logs/version_172/checkpoints/Romera-mnist-epoch=41-avg_val_ACC=0.88582-avg_val_AUPRC=0.96886-avg_val_AUROC=0.94569-avg_val_Precision=0.91569-avg_val_Recall=0.90751-avg_val_F1Score=0.91155.ckpt'
    CNN = CNN()
    checkpoint = torch.load(CNN_path)
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}
    CNN.load_state_dict(checkpoint['state_dict'], strict=False)

    # 集成模型

    # 测试直接使用拟南芥进行测试,不迁移
    model = Ensemble(RNN, BERT, CNN)
    model_path = './lightning_logs/version_0/checkpoints/ENSEM-mnist-epoch=12-avg_val_loss=0.25594-avg_val_ACC=0.89755-avg_val_AUPRC=0.97467-avg_val_AUROC=0.95504-avg_val_Precision=0.92664-avg_val_Recall=0.91465-avg_val_F1Score=0.92056.ckpt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)



    # 在这里进行数据的提取, 这里只把输入做可视化

    # 打印不出数据了救命啊()

    '''
    
    #进行迁移
    model_path = './lightning_logs/version_0/checkpoints/ENSEM-mnist-epoch=12-avg_val_loss=0.25594-avg_val_ACC=0.89755-avg_val_AUPRC=0.97467-avg_val_AUROC=0.95504-avg_val_Precision=0.92664-avg_val_Recall=0.91465-avg_val_F1Score=0.92056.ckpt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # 提取参数
    for key, value in checkpoint['state_dict'].items():
        print(f"Layer: {key}")
        print(f"   Shape: {value.shape}")
        # print(f"   Size: {value.numel()}")
        # print(f"   Parameters: {value}")

    '''

    
    # 训练器
    #trainer = pl.Trainer(accelerator="gpu", gpus=[0], max_epochs=1, logger=None)
                         # val_check_interval=0.125,
                         # callbacks=[checkpoint_callback])

    trainer = pl.Trainer(accelerator="gpu", gpus=[0], max_epochs=20,
                         val_check_interval=0.05,
                         min_epochs=20, callbacks=[checkpoint_callback])#, resume_from_checkpoint='/mnt/sdb/home/jy/contact_map/lightning_logs/version_25/checkpoints/sample-mnist-epoch=780-val_loss=2.14.ckpt')
    # trainer.fit(model, loader, loader_valid)
    # trainer = pl.Trainer(accelerator="gpu", gpus=[1], max_epochs=1100, min_epochs=1000
    # 仨模型反正是都不理想.....
    # trainer.fit(model, loader, loader_valid)  # 训练
    trainer.test(model, loader_valid)         # 测试
    #trainer.save_checkpoint("lightning_logs/example.ckpt")
    print("测试程序完成")