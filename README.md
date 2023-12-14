# CFA&RGCN

### 环境要求
```
tensorflow 
numpy 
scipy 
networkx 
torch
tqdm
```
### 用法
```
cd src/CFA
python attack.py --dataset cora --target_class_id 0 
python src/train.py --dataset cora

```
### 可选参数:

```
--dataset  数据集名称.
--learning_rate  初始化学习率.
--epochs  选择数量的epochs训练.
--hidden  选择数量的隐藏层计算单元.
--dropout Dropout率 
--para_kl L1正则化.
--early_stopping.

```
# RGCN
