from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("这里可以直接插入数据", trust_remote_code=True)

# root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/8b7d33596d18c5e83e2da052d05ca4db02e60620", trust_remote_code=True)


# '''
# cd X
# // 关于启动docker的应用。。。。

# python cli_demo.py --from_pretrained checkpoints --prompt_zh '详细描述这张胸部X光片的诊断结果'
# python cli_demo.py --from_pretrained checkpoints --prompt_zh '详细描述这张胸部X光片的诊断结果' TORCH_USE_CUDA_DSA

# 将数据权重放进docker里面，，，，，

# 1. 先把权重放到准确位置
# 2. 然后研究一下怎么部署

# 主要export CUDA_VISIBLE_DEVICES=3
# 设置在卡三上运行数据

# https://img0.baidu.com/it/u=481005968,19707510&fm=253&fmt=auto&app=138&f=JPEG?w=564&h=500


# '''

# 能跑通的位置
# ./huggingface/hub/models--THUDM--chatglm-7b/snapshots/8b7d33596d18c5e83e2da052d05ca4db02e60620
# 只要是能探索到那个模型的json就可以了

#  python cli_test.py --from_pretrained checkpoints --prompt_zh '详细描述这张胸部X光片的诊断结果'