import torch
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score,auc
import numpy as np
import time
import pandas as pd
import time

# np的模型
from Deepsignal.deepsignal2.deepsignal2.models import ModelBiLSTM

# 我的模型





# 数据格式如下, 原谅我, 倪鹏老师, 你的数据处理实在是有点抽象
# 总之这个代码绝对是反直觉的
# 这里建议使用torch.optim.Adam 优化器进行优化
# 几个指标的计算方法 :
# avg_val_loss avg_val_ACC avg_val_AUPRC avg_val_AUROC avg_val_Precision avg_val_Recall avg_val_F1Score 计算要求是.5f
# 一般是按照整个epoch进行计算, 同时统计最好情况下epoch的数据内容
'''
从csv文件中获取数据的方法
参数: @{filename}:string
返回 ['CTGAAGCGCCTGG', 
     [0.149493,0.539811,-0.293082,-0.570105,-0.855157,-0.690549,0.038542,-0.454248,0.832788,0.346973,1.467015,-0.396129,-1.722358], 
     [0.191648,0.400595,0.05592,0.113981,0.026019,0.037232,0.227943,0.162835,0.133102,0.097451,0.196424,0.156263,0.264064], 
     [17,33,3,3,6,3,5,7,7,26,10,9,3],  1]
重新補充一下: 
1. 第一個數組為電流信號, 均值, means
2. 第二個參數為电流信号波动, 标准差, std
3. 第三個為信號長度, len
'''

def get_current_time_str():
    # 获取本地时间元组
    local_time = time.localtime()
    # 格式化为字符串
    time_str = time.strftime("%Y-%m-%d(%H:%M:%S)", local_time)
    return time_str
# 自定义数据集类, 通过list获取数据
class MethylDataset(Dataset):
    def __init__(self, datas):
        self.base_to_idx = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
        self.datas = datas
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        data = self.datas[idx]
        # 特征转换（参考网页1[1](@ref)）
        kmer = torch.tensor([self.base_to_idx[b] for b in data[0]], dtype=torch.long)
        base_means = torch.tensor(data[1], dtype=torch.float32)
        base_stds = torch.tensor(data[2], dtype=torch.float32)
        base_signal_lens = torch.tensor(data[3], dtype=torch.float32)
        label = torch.nn.functional.one_hot(torch.tensor(data[4]), num_classes=2).to(torch.float)
        return kmer, base_means, base_stds, base_signal_lens, label
# 初始化数据加载器
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
datas=get_list_from_csv(dataset_tha)
train, val = sequential_split(datas)
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

dataset1 = MethylDataset(train)
dataset2 = MethylDataset(val)
dataloader = DataLoader(dataset1, batch_size=100, shuffle=True, num_workers=0)
val_dataloader = DataLoader(dataset2, batch_size=100, shuffle=True, num_workers=0)

# 设备配置（参考网页1[1](@ref)）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型==============================================================================================
# 我的模型


# nipeng的模型
model = ModelBiLSTM(
    module="seq_bilstm",
    is_signallen=True,
    vocab_size=16,
    seq_len=13,
    device=0
).to(device)


# 设置优化器学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss() # 这里应该使用二元的交叉熵损失函数


# 设置AUROC作为最终评价指标, 获取模型的参数信息
max_AUROC=0
weight_message="epoch={}, loss={}, AUROC={}"
weights= model.state_dict()
execution_minutes=0
sum_min=0



# 训练循环
for epoch in range(25):  # 训练25个epoch
    model.train() # 切换为训练模式
    running_loss = 0.0
    i=0
    # 记录起始时间戳
    start_time = time.time()
    for batch in dataloader:


        print(f'In training Epoch [{epoch + 1}], process: {100*i}/{len(train)}')
        i=i+1
        # 数据加载
        kmer, base_means, base_stds, base_signal_lens, labels = batch
        signals = torch.zeros(kmer.size(0), 13, 16)  # 信号占位符

        # 设备转移
        inputs = [t.to(device) for t in (kmer, base_means, base_stds, base_signal_lens, signals)]
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs, prob = model(*inputs)
        loss = criterion(prob, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}], Loss: {running_loss / len(dataloader):.4f}')

    model.eval()
    val_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for batch in val_dataloader:
            kmer, base_means, base_stds, base_signal_lens, labels = batch
            signals = torch.zeros(kmer.size(0), 13, 16)

            # 设备转移
            inputs = [t.to(device) for t in (kmer, base_means, base_stds, base_signal_lens, signals)]
            labels = labels.to(device)

            # 前向传播
            outputs, prob = model(*inputs)
            loss = criterion(prob, labels)

            # 累积数据（保持原始维度）[1,3](@ref)
            val_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())  # 形状 [n,2] 的 one-hot 编码
            all_probs.extend(prob.cpu().numpy())  # 形状 [n,2] 的概率值
    # 记录结束时间戳
    end_time = time.time()

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


    if avg_val_AUROC>max_AUROC:
        print("检测到权重更新并且进行保存")
        #max_AUROC = avg_val_AUROC
        #weights=model.state_dict()


#torch.save(model.state_dict(), f'Start={get_current_time_str()}_Time={execution_minutes}_{weight_message}.pth')
print("训练完成")



