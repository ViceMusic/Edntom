import torch
from Edntom.BERT_plus_I import BERT_plus_encoder
from Edntom.RNN_I import biRNN_basic
from Edntom.CNN_rep import CNN
from Edntom.Ensemble import Ensemble
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score,auc

# 对于list的处理模块
def process_multimodal_data(datas):
    data_list=[]
    for data in datas:
        kmer=data[0]
        combined = [
            [data[1][i], data[2][i], data[3][i]]
            for i in range(len(data[1]))  # 假设三个数组长度均为13（网页4的维度校验）
        ]
        label=data[4]
        data_list.append([kmer,combined,label])

    # 处理三维特征（综合网页2/3/4的多维数组操作）


    return data_list
# 新建dataset类


class MultiInputDataset(Dataset):
    def __init__(self, datas):
        self.base_to_idx = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
        self.datas = datas
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        data = self.datas[idx]
        # 特征转换（参考网页1[1](@ref)）
        kmer = data[0]
        nano_data = torch.tensor(data[1], dtype=torch.float32)
        label = torch.nn.functional.one_hot(torch.tensor(data[2]), num_classes=2).to(torch.float)
        return kmer, nano_data, label





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
# BERT_path = './lightning_logs/version_60/checkpoints/ENSEM-mnist-epoch=24-avg_val_loss=0.21958-avg_val_ACC=0.94567-avg_val_AUPRC=0.98672-avg_val_AUROC=0.98643-avg_val_Precision=0.95039-avg_val_Recall=0.94059-avg_val_F1Score=0.94541.ckpt'
# 拟南芥上训练的
# BERT_path = './lightning_logs/version_59/checkpoints/ENSEM-mnist-epoch=24-avg_val_loss=0.22529-avg_val_ACC=0.94172-avg_val_AUPRC=0.97810-avg_val_AUROC=0.98159-avg_val_Precision=0.92108-avg_val_Recall=0.96673-avg_val_F1Score=0.94331.ckpt'

BERT = BERT_plus_encoder(vocab_size=7, hidden=100, n_layers=3, attn_heads=4, dropout=0).float()
checkpoint = torch.load(BERT_path)
checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'linear' not in k}
BERT.load_state_dict(checkpoint['state_dict'], strict=False)

# 人类数据集上训练的
CNN_path = '/mnt/sde/ycl/NanoCon/code/lightning_logs/version_172/checkpoints/Romera-mnist-epoch=41-avg_val_ACC=0.88582-avg_val_AUPRC=0.96886-avg_val_AUROC=0.94569-avg_val_Precision=0.91569-avg_val_Recall=0.90751-avg_val_F1Score=0.91155.ckpt'
# 水稻数据集上训练的
# CNN_path = './lightning_logs/version_62/checkpoints/ENSEM-mnist-epoch=24-avg_val_loss=0.14277-avg_val_ACC=0.94751-avg_val_AUPRC=0.98775-avg_val_AUROC=0.98754-avg_val_Precision=0.95180-avg_val_Recall=0.94268-avg_val_F1Score=0.94717.ckpt'
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
checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}
model.load_state_dict(checkpoint['state_dict'], strict=False)



print("数据加载完成")

import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd

# 获取时间的方法
def get_current_time_str():
    # 获取本地时间元组
    local_time = time.localtime()
    # 格式化为字符串
    time_str = time.strftime("%Y-%m-%d(%H:%M:%S)", local_time)
    return time_str


# 用户平衡数据, 并在张量阶段就划分出测试和训练
def sequential_split(datas):
    # 按标签分层存储
    # 按标签分层存储
    label_0 = []
    label_1 = []
    for item in datas:
        if item[4] == 0:  # 假设标签在第4位
            label_0.append(item)
        else:
            label_1.append(item)
    groups = [label_0, label_1]  # 手动分桶代替哈希

    # 第二阶段：各层独立划分
    train_set, val_set = [], []
    for group in groups:
        if not group:  # 空组跳过
            continue

        total = len(group)
        split_point = int(total * 0.9)

        # 处理极小样本（参考网页3的非比例抽样思想）
        if split_point == 0 and total >= 1:  # 至少保留1个样本到验证集
            split_point = max(0, total - 1)
        elif split_point == total:  # 防止全量划入训练集
            split_point = total - 1

        train_set.extend(group[:split_point])
        val_set.extend(group[split_point:])

    # 全局随机重组（参考网页4的灵活性原则）
    random.shuffle(train_set)
    random.shuffle(val_set)
    return train_set, val_set

# 调用示例
dataset_human = "/mnt/sde/tiange/Nanopore_data/humen1.csv" # 人类数据集
dataset_tha = "~/Ensemble/data/balance_tha.csv"  # 拟南芥数据集
dataset_rice = "~/Ensemble/data/balance_rice.csv"  # 水稻数据集


# 获取数据, 划分集合
def get_list_from_csv(filename):
    # 内置一个转化方法
    def deep_convert(item):
        if isinstance(item, list):
            return [deep_convert(x) for x in item]
        elif isinstance(item, str) and "," in item:
            return [float(x) if "." in x else int(x) for x in item.split(",")]
        else:
            return item
    # 读取 CSV 文件
    df = pd.read_csv(filename)  # 替换为你的文件路径
    # 转换为列表（每行一个子列表）
    data_list = df.values.tolist()
    data_list=deep_convert(data_list)
    # 输出前 5 行
    print("前 5 行数据：")
    for row in data_list[:5]:
        print(row)
    return data_list

# 使用这个玩意也太抽象了
data = torch.load('methylation_datasets.pth')
train_set = data['train']
val_set = data['val']
train, val = sequential_split(train_set)
print(f"训练集大小: {len(train)}, 验证集大小: {len(val)}")

# 输出：训练集大小:90, 验证集大小:10
t=0
f=0
for i in range(len(val)):
    if val[i][4]==1:
        t=t+1
    else:
        f=f+1
print("检查数据格式",t,f)



train = process_multimodal_data(train)
print("检查",torch.Tensor(train[1][1]).shape)
dataset1 = MultiInputDataset(train)
val = process_multimodal_data(val)
dataset2 = MultiInputDataset(val)
dataloader = DataLoader(dataset1, batch_size=1000, shuffle=True, num_workers=4)
val_dataloader = DataLoader(dataset1, batch_size=2000, shuffle=True, num_workers=4)

# 设备配置（参考网页1[1](@ref)）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Ensemble(RNN, BERT, CNN).to(device)


# 设置优化器学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss() # 这里应该使用二元的交叉熵损失函数
import time
# 设置AUROC作为最终评价指标, 获取模型的参数信息
max_AUROC=0
weight_message="epoch={}, loss={}, AUROC={}"
weights= model.state_dict()
execution_minutes=0
sum_min=0

with torch.no_grad():
    for batch in val_dataloader:
        kmer, nano_data, labels = batch

        torch.save(labels,"label.pth")
        exit("暂停")

        # 设备转移

        labels = labels.to(device)
        nano_data = nano_data.to(device)

        # 前向传播
        prob = model(kmer, nano_data)









# 训练循环
for epoch in range(30):  # 训练25个epoch
    model.train() # 切换为训练模式
    running_loss = 0.0
    i=0
    # 记录起始时间戳
    start_time = time.time()
    for batch in dataloader:
        i=i+1
        # 数据加载
        kmer, nano_data, labels = batch

        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        prob = model(kmer, nano_data.to(device))
        prob = torch.softmax(prob, dim=1)  # 方式2：直接调用Tensor方法
        loss = criterion(prob, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'epoch [{epoch + 1}], Loss: {running_loss / len(dataloader):.4f}')
    # 记录结束时间戳
    end_time = time.time()

    model.eval()
    val_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for batch in val_dataloader:
            kmer, nano_data, labels = batch

            # 设备转移

            labels = labels.to(device)
            nano_data=nano_data.to(device)

            # 前向传播
            prob = model(kmer, nano_data)
            prob = torch.softmax(prob, dim=1)  # 方式2：直接调用Tensor方法
            loss = criterion(prob, labels)

            # 累积数据（保持原始维度）[1,3](@ref)
            val_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())  # 形状 [n,2] 的 one-hot 编码
            all_probs.extend(prob.cpu().numpy())  # 形状 [n,2] 的概率值


    import numpy as np

    # 格式转换
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 转换为类别索引（针对 one-hot 和多分类）[3,4](@ref)
    y_true = np.argmax(all_labels, axis=1)  # 从 one-hot 转为一维类别标签（0或1）
    y_pred = np.argmax(all_probs, axis=1)  # 取概率最大的类别索引

    # 计算分类指标（统一用类别索引格式）[1,4](@ref)
    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_ACC = accuracy_score(y_true, y_pred)
    avg_val_Precision = precision_score(y_true, y_pred, average='binary')  # 二分类需指定average
    avg_val_Recall = recall_score(y_true, y_pred, average='binary')
    avg_val_F1Score = f1_score(y_true, y_pred, average='binary')

    # 计算概率指标（使用正类概率）[4,5](@ref)
    positive_probs = all_probs[:, 1]  # 取第二列（正类）概率
    avg_val_AUROC = roc_auc_score(y_true, positive_probs)
    precision, recall, _ = precision_recall_curve(y_true, positive_probs)
    avg_val_AUPRC = auc(recall, precision)

    # 计算分钟数（绝对值保证正数）
    execution_minutes = abs(end_time - start_time) / 60

    # 打印验证指标
    print(f"{epoch}轮的验证效果为:======================================================")
    print(f"本轮次代码执行耗时: {execution_minutes:.2f} 分钟")
    print(f'Val_Loss: {avg_val_loss:.5f} | '
          f'ACC: {avg_val_ACC:.5f} | '
          f'AUROC: {avg_val_AUROC:.5f} | '
          f'AUPRC: {avg_val_AUPRC:.5f}\n'
          f'Precision: {avg_val_Precision:.5f} | '
          f'Recall: {avg_val_Recall:.5f} | '
          f'F1: {avg_val_F1Score:.5f}')
    weight_message = f"epoch={epoch}_Val_Loss={avg_val_loss:.5f}_ACC={avg_val_ACC:.5f}_AUROC={avg_val_AUROC:.5f}_AUPRC={avg_val_AUPRC:.5f}_Precision={avg_val_Precision:.5f}_Recall={avg_val_Recall:.5f}_F1={avg_val_F1Score:.5f}"

    # 计算分钟数（绝对值保证正数）
    execution_minutes = abs(end_time - start_time) / 60
    print(f"本轮次代码执行耗时: {execution_minutes:.2f} 分钟")

    if avg_val_AUROC>max_AUROC:
        print("检测到权重更新并且进行保存")
        max_AUROC = avg_val_AUROC
        weights=model.state_dict()



#torch.save(model.state_dict(), f'EDNTOM_Start={get_current_time_str()}_Time={execution_minutes}_{weight_message}.pth')
print("训练完成")
