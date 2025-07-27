import torch
from pytorch_lighting_model.BERT_plus_I import BERT_plus_encoder
from pytorch_lighting_model.RNN_I import biRNN_basic
from pytorch_lighting_model.CNN_rep import CNN
from pytorch_lighting_model.Ensemble import Ensemble
from utils import load_dataset1, make_data, MyDataSet
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl


if __name__ == '__main__':

    #data_load = load_dataset1("/mnt/sde/tiange/Nanopore_data/humen1.csv") # 人类数据集
    #data_load = load_dataset1("./data/balance_tha.csv")  # 拟南芥数据集
    data_load = load_dataset1("./data/balance_rice.csv") # 水稻数据集

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
    loader = DataLoader(mnist_train, 100, True)#, collate_fn=collate)#, num_workers=40)#, collate_fn=custom_collate_fn)#, num_workers=12)
    loader_valid = DataLoader(mnist_val, 2000, False)#, collate_fn=collate)#, num_workers=12) #, num_workers=32)#, collate_fn=custom_collate_fn)#,num_workers=12)
    #loader_test = DataLoader(mnist_test, 2000, False) #232

    #在测试数据集上选取2000数据
    #数据来源为水稻数据集
    #loader_test = DataLoader(mnist_val, 2000, False)

    # 人类数据集上训练的
    RNN_path = '/mnt/sde/ycl/NanoCon/code/lightning_logs/version_163/checkpoints/RNN-mnist-epoch=49-avg_val_ACC=0.86791-avg_val_AUPRC=0.95966-avg_val_AUROC=0.93144-avg_val_Precision=0.89477-avg_val_Recall=0.90262-avg_val_F1Score=0.89864.ckpt'
    # 水稻数据集上训练的
    # RNN_path = './lightning_logs/version_57/checkpoints/ENSEM-mnist-epoch=24-avg_val_loss=0.19370-avg_val_ACC=0.92298-avg_val_AUPRC=0.97655-avg_val_AUROC=0.97724-avg_val_Precision=0.91133-avg_val_Recall=0.93755-avg_val_F1Score=0.92418.ckpt'
    # 拟南芥数据上训练的
    # RNN_path = './lightning_logs/version_58/checkpoints/ENSEM-mnist-epoch=24-avg_val_loss=0.19593-avg_val_ACC=0.93179-avg_val_AUPRC=0.96566-avg_val_AUROC=0.97287-avg_val_Precision=0.91417-avg_val_Recall=0.95404-avg_val_F1Score=0.93361.ckpt'

    RNN = biRNN_basic()
    checkpoint = torch.load(RNN_path)
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}
    RNN.load_state_dict(checkpoint['state_dict'], strict=False)

    # 人类数据集上训练的
    BERT_path = '/mnt/sde/ycl/NanoCon/code/lightning_logs/version_166/checkpoints/BERT-mnist-epoch=47-avg_val_ACC=0.88783-avg_val_AUPRC=0.96826-avg_val_AUROC=0.94672-avg_val_Precision=0.92553-avg_val_Recall=0.89940-avg_val_F1Score=0.91225.ckpt'
    # 水稻数据集上训练的
    #BERT_path = './lightning_logs/version_60/checkpoints/ENSEM-mnist-epoch=24-avg_val_loss=0.21958-avg_val_ACC=0.94567-avg_val_AUPRC=0.98672-avg_val_AUROC=0.98643-avg_val_Precision=0.95039-avg_val_Recall=0.94059-avg_val_F1Score=0.94541.ckpt'
    # 拟南芥上训练的
    # BERT_path = './lightning_logs/version_59/checkpoints/ENSEM-mnist-epoch=24-avg_val_loss=0.22529-avg_val_ACC=0.94172-avg_val_AUPRC=0.97810-avg_val_AUROC=0.98159-avg_val_Precision=0.92108-avg_val_Recall=0.96673-avg_val_F1Score=0.94331.ckpt'

    BERT = BERT_plus_encoder(vocab_size=7, hidden=100, n_layers=3, attn_heads=4, dropout=0).float()
    checkpoint = torch.load(BERT_path)
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'linear' not in k}
    BERT.load_state_dict(checkpoint['state_dict'], strict=False)

    # 人类数据集上训练的
    CNN_path = '/mnt/sde/ycl/NanoCon/code/lightning_logs/version_172/checkpoints/Romera-mnist-epoch=41-avg_val_ACC=0.88582-avg_val_AUPRC=0.96886-avg_val_AUROC=0.94569-avg_val_Precision=0.91569-avg_val_Recall=0.90751-avg_val_F1Score=0.91155.ckpt'
    # 水稻数据集上训练的
    #CNN_path = './lightning_logs/version_62/checkpoints/ENSEM-mnist-epoch=24-avg_val_loss=0.14277-avg_val_ACC=0.94751-avg_val_AUPRC=0.98775-avg_val_AUROC=0.98754-avg_val_Precision=0.95180-avg_val_Recall=0.94268-avg_val_F1Score=0.94717.ckpt'
    # 拟南芥数据集上训练的
    # CNN_path = './lightning_logs/version_63/checkpoints/ENSEM-mnist-epoch=16-avg_val_loss=0.17215-avg_val_ACC=0.93753-avg_val_AUPRC=0.97667-avg_val_AUROC=0.98003-avg_val_Precision=0.92573-avg_val_Recall=0.95284-avg_val_F1Score=0.93905.ckpt'

    CNN = CNN()
    checkpoint = torch.load(CNN_path)
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}
    CNN.load_state_dict(checkpoint['state_dict'], strict=False)

    # 集成模型
    # 测试直接使用拟南芥进行测试,不迁移
    model = Ensemble(RNN, BERT, CNN)
    # 在人类数据集上训练的模型
    model_path = './lightning_logs/version_0/checkpoints/ENSEM-mnist-epoch=12-avg_val_loss=0.25594-avg_val_ACC=0.89755-avg_val_AUPRC=0.97467-avg_val_AUROC=0.95504-avg_val_Precision=0.92664-avg_val_Recall=0.91465-avg_val_F1Score=0.92056.ckpt'
    # 在水稻数据集上迁移的模型
    # model_path = './lightning_logs/version_16/checkpoints/ENSEM-mnist-epoch=43-avg_val_loss=0.14033-avg_val_ACC=0.94730-avg_val_AUPRC=0.98801-avg_val_AUROC=0.98760-avg_val_Precision=0.95109-avg_val_Recall=0.94321-avg_val_F1Score=0.94709.ckpt'
    # 在拟南芥数据集上训练的模型
    # model_path = './lightning_logs/version_17/checkpoints/ENSEM-mnist-epoch=29-avg_val_loss=0.13023-avg_val_ACC=0.94922-avg_val_AUPRC=0.98937-avg_val_AUROC=0.98914-avg_val_Precision=0.94655-avg_val_Recall=0.95291-avg_val_F1Score=0.94968.ckpt'

    # 上面是迁移, 下面才是重新训练的结果
    # 在水稻上面训练的
    # model_path='./lightning_logs/version_65/checkpoints/ENSEM-mnist-epoch=13-avg_val_loss=0.10854-avg_val_ACC=0.95897-avg_val_AUPRC=0.99267-avg_val_AUROC=0.99253-avg_val_Precision=0.96458-avg_val_Recall=0.95308-avg_val_F1Score=0.95876.ckpt'
    # 在拟南芥上面训练的
    # model_path='./lightning_logs/version_66/checkpoints/ENSEM-mnist-epoch=22-avg_val_loss=0.12601-avg_val_ACC=0.95147-avg_val_AUPRC=0.98878-avg_val_AUROC=0.98953-avg_val_Precision=0.94130-avg_val_Recall=0.96309-avg_val_F1Score=0.95202.ckpt'

    checkpoint = torch.load(model_path)
    #model.load_state_dict(checkpoint['state_dict'], strict=False)

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

    trainer = pl.Trainer(accelerator="gpu", gpus=[0], max_epochs=1,
                         val_check_interval=0.05,
                         min_epochs=1, callbacks=[checkpoint_callback])#, resume_from_checkpoint='/mnt/sdb/home/jy/contact_map/lightning_logs/version_25/checkpoints/sample-mnist-epoch=780-val_loss=2.14.ckpt')

    #trainer.fit(model, loader, loader_valid)


    # 计算总参数数量
    #total_params = sum(p.numel() for p in model.parameters())
    # print(f"总参数量：{total_params / 1e6:.2f}M")  # 转换为百万单位
    # 总参数数量为1.77M
    # 计算不可训练的参数数量


    # trainer = pl.Trainer(accelerator="gpu", gpus=[1], max_epochs=1100, min_epochs=1000
    # trainer.fit(RNN, loader, loader_valid)  # 训练
    #print("测试读取")
    #print('EDNTOM========')
    #trainer.test(model, loader_valid)         # 测试
    #print('RNN===========')
    #trainer.test(RNN, loader_valid)  # 测试
    #print('BERT==========')
    #trainer.test(BERT, loader_valid)  # 测试
    #print('CNN===========')
    trainer.test(model, loader_valid)  # 测试
    #trainer.save_checkpoint("lightning_logs/example.ckpt")

    #在这里吧四个模型都验证
    # 印象里应该是不用再继续训练了