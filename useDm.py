# nohup python3 useDm.py > dm.log 2> dm_err.log &
# nvidia-smi

import tensorflow as tf
from tensorflow.contrib import rnn
import random
import pandas as pd

# 指定使用序号为0的GPU（网页8、9、10）
import os


# 替代方案二：获取设备列表（旧版实验性 API）
from tensorflow.python.client import device_lib
def get_devices():
    return [x.name for x in device_lib.list_local_devices() if x.device_type == 'XLA_GPU']
print("GPU List:", get_devices())  # 输出示例：['/device:GPU:0'][7](@ref)

# 指定物理设备ID（例如选择第0号卡）
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"  # 对应物理卡ID，非逻辑顺序[8](@ref)
config.gpu_options.allow_growth = True  # 启用显存动态分配[8](@ref)


import numpy as np

import math
import glob, os, sys, time;

from collections import defaultdict


# different class weights for unbalanced data
class_weights = tf.constant([0.1,0.9])

#
# create a RNN with LSTM
# define performance evaluation operation
#
def mCreateSession(num_input, num_hidden, timesteps, moptions):
   # two classes only
   num_classes = 2;
   # the number of layers
   numlayers = 3;
   # learning rate
   learning_rate = 0.001

   # define input and output
   X = tf.placeholder("float", [None, timesteps, num_input]);
   Y = tf.placeholder("float", [None, num_classes]);

   # for last layers
   weights = {'out': tf.Variable(tf.truncated_normal([2*num_hidden, num_classes]))};
   biases = {'out': tf.Variable(tf.truncated_normal([num_classes]))}

   # define a bidirectional RNN
   def BiRNN(x, weights, biases):
      x = tf.unstack(x, timesteps, 1);

      # define the LSTM cells
      lstm_fw_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) for _ in range(numlayers)]);
      lstm_bw_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) for _ in range(numlayers)]);

      # define bidirectional RNN
      try:
         outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32);
      except Exception:
         outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32);

      # define output layer
      if moptions['outputlayer'] in ['sigmoid']:
         return tf.contrib.layers.fully_connected(outputs[int(timesteps/2)], num_outputs=num_classes, activation_fn=tf.nn.sigmoid);
      else:
         return tf.matmul(outputs[int(timesteps/2)], weights['out']) + biases['out']

   # get prediction
   logits = BiRNN(X, weights, biases);
   prediction = tf.nn.softmax(logits)

   mfpred=tf.argmax(prediction,1)

   ## with different class-weights or not
   if 'unbalanced' in moptions and (not moptions['unbalanced']==None) and moptions['unbalanced']==1:  # class_weights
      loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.multiply(logits, class_weights), labels=Y))
   else:
      loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
   #

   # for optimizer
   optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate);
   train_op = optimizer.minimize(loss_op);

   # get accuracy
   correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1));
   accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32));

   # AUC
   auc_op = tf.metrics.auc(Y, prediction)
   # precision
   mpre = tf.metrics.precision(tf.argmax(Y, 1), tf.argmax(prediction, 1))
   # recall
   mspf = tf.metrics.recall(tf.argmax(Y, 1), tf.argmax(prediction, 1))

   # initialization of variables
   init = tf.global_variables_initializer();
   init_l = tf.local_variables_initializer()

   saver = tf.train.Saver();

   # 检查参数数目
   param_counter = tf.trainable_variables()
   total_params = np.sum([np.prod(v.get_shape().as_list()) for v in param_counter])

   print("Total trainable parameters:", total_params)


   return (init, init_l, loss_op, accuracy, train_op, X, Y, saver, auc_op, mpre, mspf, mfpred)




import numpy as np

# 处理数据的方法
import numpy as np


def convert_to_onehot(datas):
   """
   批量转换DNA序列和信号数据为one-hot编码格式
   参数：
       datas : list of tuples
           输入数据列表，每个元素为(dna_seq, signals, label)
           dna_seq: str类型，DNA序列
           signals: list/array，信号强度数组
           label: int类型，分类标签
   返回：
       features_array : numpy数组，形状为[样本数, 序列长度, 5]
       labels_array : numpy数组，形状为[样本数, 2]
   """
   # 定义增强版碱基字典（参考网页5）
   base_dict = {
      'A': [1, 0, 0, 0], 'a': [1, 0, 0, 0],
      'G': [0, 1, 0, 0], 'g': [0, 1, 0, 0],
      'C': [0, 0, 1, 0], 'c': [0, 0, 1, 0],
      'T': [0, 0, 0, 1], 't': [0, 0, 0, 1],
      'N': [0.25, 0.25, 0.25, 0.25]  # 处理模糊碱基（网页1扩展）
   }

   features_list = []
   labels_list = []

   for data in datas:
      # 数据解包校验（参考网页6）
      try:
         dna_seq, signals,_ , _ , label = data
         assert len(dna_seq) == len(signals), "序列与信号长度不匹配"
      except (ValueError, AssertionError) as e:
         print(f"error: {e}")
         continue

      # 特征生成（网页4方法增强）
      sample_features = []
      for i, base in enumerate(dna_seq):
         try:
            base_encoding = base_dict[base]
         except KeyError:
            base_encoding = base_dict['N']  # 处理未知碱基

         # 拼接信号特征（网页1方法扩展）
         feature_vector = base_encoding + [signals[i]]
         sample_features.append(feature_vector)

      # 标签编码（参考网页2）
      label_encoding = [1, 0] if label == 0 else [0, 1]

      features_list.append(sample_features)
      labels_list.append(label_encoding)

   # 转换为numpy数组（网页1最佳实践）
   return features_list, labels_list

'''
接下来开始手动跑这个模型了()

'''


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
dataset_human = "/mnt/sde/tiange/Nanopore_data/humen1.csv"  # 人类数据集
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
   data_list = deep_convert(data_list)
   return data_list


# 使用这个玩意也太抽象了
datas = get_list_from_csv(dataset_tha)
train, val = sequential_split(datas)
print(f"train: {len(train)}, valid: {len(val)}")
# 输出：训练集大小:90, 验证集大小:10
t = 0
f = 0
for i in range(len(val)):
   if val[i][4] == 1:
      t = t + 1
   else:
      f = f + 1
print("chech the dist", t, f)



# 将数据转化为模型需要的格式
features_train,labels_train=convert_to_onehot(train)
features_val,labels_val=convert_to_onehot(val)


# 先将其转化为肯恶搞会用到的数组
features_train = np.array(features_train, dtype=np.float32)  # 转换为[n,13,5]的数组
labels_train = np.array(labels_train, dtype=np.float32)      # 形状为[n, 2]

features_val = np.array(features_val, dtype=np.float32)  # 转换为[n,13,5]的数组
labels_val = np.array(labels_val, dtype=np.float32)      # 形状为[n, 2]



# 无需标准化数组, 设置一些数据类型
num_input = 5    # 特征维度（4碱基+13电信号means）
num_hidden = 64  # LSTM隐层单元数
timesteps = 13    # 时间步长
moptions = {'outputlayer': None}  # 不使用Sigmoid输出层
# 获取模型组件
import tensorflow as tf
# 获取模型组件
(init, init_l, loss_op, accuracy, train_op, X, Y, _, auc_op, mpre, mspf, mfpred) = \
    mCreateSession(num_input, num_hidden, timesteps, moptions)
# One-hot标签：形状 (1, 2)

# 原始代码基础上添加训练循环
with tf.Session(config=config) as sess:
   # 初始化所有变量（使用函数返回的init和init_l）
   sess.run([init, init_l])  # init和init_l来自mCreateSession返回值


   # 数据参数
   train_samples = features_train.shape[0]
   val_samples = features_val.shape[0]
   batch_size = 100
   n_epochs = 25

   for epoch in range(n_epochs):
      # === 训练阶段 ===
      epoch_loss = 0.0
      epoch_acc = 0.0
      total_batch = train_samples // batch_size

      start_time = time.time()
      for i in range(total_batch):
         print(f"epoch[{epoch}]:{i*batch_size}/{train_samples}")
         # 切片获取训练批次（无新变量）
         batch_x = features_train[i * batch_size: (i + 1) * batch_size]
         batch_y = labels_train[i * batch_size: (i + 1) * batch_size]

         # 执行训练操作（仅使用函数返回的train_op/loss_op/accuracy）
         _, loss_val, acc_val = sess.run(
            [train_op, loss_op, accuracy],
            feed_dict={X: batch_x, Y: batch_y}
         )

         epoch_loss += loss_val / total_batch
         epoch_acc += acc_val / total_batch

      # === 验证阶段 ===
      # 重置局部指标状态（使用函数返回的init_l）
      sess.run(init_l)

      # 累积验证指标（直接使用函数返回的auc_op/mpre/mspf）
      val_metrics = {
         'loss': [], 'acc': [], 'auc': [],
         'precision': [], 'recall': []
      }

      for i in range(val_samples // batch_size):
         # 切片获取验证批次（无新变量）
         val_x = features_val[i * batch_size: (i + 1) * batch_size]
         val_y = labels_val[i * batch_size: (i + 1) * batch_size]

         # 获取所有指标（严格使用函数返回的操作）
         loss_val, acc_val, auc_val, pre_val, rec_val = sess.run(
            [loss_op, accuracy, auc_op[1], mpre[1], mspf[1]],
            feed_dict={X: val_x, Y: val_y}
         )

         val_metrics['loss'].append(loss_val)
         val_metrics['acc'].append(acc_val)
         val_metrics['auc'].append(auc_val)
         val_metrics['precision'].append(pre_val)
         val_metrics['recall'].append(rec_val)

      # 计算均值（动态生成中间变量）
      avg_val_loss = np.mean(val_metrics['loss'])
      avg_val_acc = np.mean(val_metrics['acc'])
      avg_val_auc = np.mean(val_metrics['auc'])
      avg_val_pre = np.mean(val_metrics['precision'])
      avg_val_rec = np.mean(val_metrics['recall'])
      avg_val_f1 = 2 * (avg_val_pre * avg_val_rec) / (avg_val_pre + avg_val_rec + 1e-7)
      end_time = time.time()
      execution_minutes = abs(end_time - start_time) / 60
      print(f"====================================epoch:[{epoch}]========================================")
      print(f"The consumption of this epoch: {execution_minutes:.2f} minutes")
      # 输出结果（直接格式化打印）
      print(f'Val_Loss: {avg_val_loss:.5f} | '
            f'ACC: {avg_val_acc:.5f} | '
            f'AUROC: {avg_val_auc:.5f}\n'
            f'Precision: {avg_val_pre:.5f} | '
            f'Recall: {avg_val_rec:.5f} | '
            f'F1: {avg_val_f1:.5f}')
      print("\n")