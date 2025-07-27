# 这是一个用来比较各种乱七八糟的东西的文件, 主要用于后续的对比, 之类的东西
import pandas as pd
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
# 存储三个数据集
dataset_human = "/mnt/sde/tiange/Nanopore_data/humen1.csv" # 人类数据集
dataset_tha = "./data/balance_tha.csv"  # 拟南芥数据集
dataset_rice = "./data/balance_rice.csv"  # 水稻数据集

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

'''
将csv数据转化成fasta格式
'''





# 测试区域

# data_csv=get_list_from_csv(dataset_human)
data_csv=get_list_from_csv(dataset_tha)
# data_csv=get_list_from_csv(dataset_rice)
print("成功获取数据",len(data_csv))

# 输出：训练集大小:90, 验证集大小:10
t=0
f=0
for i in range(len(data_csv)):
    if data_csv[i][4]==1:
        t=t+1
    else:
        f=f+1
print("检查数据格式",t,f)

# 人类数据集 3865931 2509760 1356171
# 拟南芥 316109 158551 157558
# 水稻 1275390 637695 637695

# 60%-80%	20%-40%	Illumina甲基化芯片数据显示, 我们的人类数据集已经轻度不平衡了
# 为了满足小馋猫, 我们把这个比例调整到5:1, 接近高度不平衡了, 一般需要采样策略

import torch
import random
import numpy as np
from collections import defaultdict


def create_imbalanced_datasets(full_data, train_size=30000, val_size=3000, imbalance_ratio=5, seed=42):
    """
    从全量数据中创建非平衡训练集和平衡验证集

    参数:
        full_data: 原始数据 (list of lists, 每个子列表的第4索引是标签)
        train_size: 训练集总样本数
        val_size: 验证集总样本数 (将强制1:1平衡)
        imbalance_ratio: 训练集的多数类:少数类比例 (如5:1)
        seed: 随机种子
    """
    # 设置随机种子保证可复现
    random.seed(seed)
    np.random.seed(seed)

    # 1. 按标签分层存储
    label_dict = defaultdict(list)
    for item in full_data:
        label = item[4]  # 假设标签在第4位
        label_dict[label].append(item)

    # 2. 计算训练集的样本分配
    minority_class = 1 if len(label_dict[1]) < len(label_dict[0]) else 0
    majority_class = 1 - minority_class

    # 训练集样本数计算 (5:1比例)
    n_train_minority = int(train_size / (imbalance_ratio + 1))
    n_train_majority = train_size - n_train_minority

    # 验证集样本数计算 (1:1比例)
    n_val_each = val_size // 2

    # 3. 采样数据 (防止样本不足)
    def safe_sample(data, n_samples):
        return random.sample(data, min(n_samples, len(data)))

    # 训练集采样
    train_majority = safe_sample(label_dict[majority_class], n_train_majority)
    train_minority = safe_sample(label_dict[minority_class], n_train_minority)

    # 验证集采样 (强制平衡)
    val_majority = safe_sample(
        [x for x in label_dict[majority_class] if x not in train_majority],
        n_val_each
    )
    val_minority = safe_sample(
        [x for x in label_dict[minority_class] if x not in train_minority],
        n_val_each
    )

    # 4. 合并数据集
    train_set = train_majority + train_minority
    val_set = val_majority + val_minority

    # 打乱顺序
    random.shuffle(train_set)
    random.shuffle(val_set)

    # 5. 统计信息
    train_stats = {
        'total': len(train_set),
        f'class_{majority_class}': len(train_majority),
        f'class_{minority_class}': len(train_minority),
        'imbalance_ratio': len(train_majority) / len(train_minority)
    }

    val_stats = {
        'total': len(val_set),
        f'class_{majority_class}': len(val_majority),
        f'class_{minority_class}': len(val_minority),
        'balance_ratio': 1.0
    }

    print(
        f"训练集创建完成: 总数={train_stats['total']}, {majority_class}:{minority_class}={train_stats[f'class_{majority_class}']}:{train_stats[f'class_{minority_class}']} (比例={train_stats['imbalance_ratio']:.1f}:1)")
    print(
        f"验证集创建完成: 总数={val_stats['total']}, {majority_class}:{minority_class}={val_stats[f'class_{majority_class}']}:{val_stats[f'class_{minority_class}']} (比例=1:1)")

    return train_set, val_set


# 创建数据集
train_data, val_data = create_imbalanced_datasets(data_csv)

# 保存为pth文件
torch.save({
    'train': train_data,
    'val': val_data,
    'metadata': {
        'train_imbalance_ratio': 5.0,
        'val_balance_ratio': 1.0
    }
}, 'methylation_datasets.pth')

print("数据集已保存为 methylation_datasets.pth")


