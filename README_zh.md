# PhaseConeGNN

[English README](README.md)

PhaseConeGNN 是一种面向 And-Inverter Graph（AIG）表示学习的相位感知逻辑锥图神经网络。该实现支持信号概率预测、等效门识别以及基于真值表距离的节点表示学习。

本仓库按照匿名评审代码包的形式整理，不包含作者身份、单位名称、个人邮箱或非匿名项目链接。

## 方法概述

PhaseConeGNN 以带符号 AIG 为输入，同时学习节点级嵌入表示和信号概率预测结果。带符号边用于区分相位保持传播和反相传播。模型主要包含以下设计：

- 为每个节点初始化原相位状态和反相位状态；
- 在正边和反相边上执行带符号相位转换消息传递；
- 使用图上下文条件归一化和门控残差更新提升深层传播稳定性；
- 对不同传播深度的节点表示进行自适应融合；
- 在信号概率读出阶段引入 signed-fanin 逻辑先验。

对于等效门识别任务，模型使用两个门节点最终嵌入之间的余弦相似度作为 gate-pair 功能相似性估计。

## 仓库结构

```text
PhaseConeGNN/
  README.md            # 英文说明文档
  README_zh.md         # 中文说明文档
  environment.yml      # Conda 环境配置
  data/
    demo_AIGDataset/   # 用于 smoke test 的小型处理后数据集
    README.md          # 数据说明
  train.py             # 训练、验证、测试与模型保存入口
  model.py             # PhaseConeGNN 主模型
  layers.py            # 相位转换、残差、融合和读出层
  normalization.py     # 图上下文条件归一化
  load_data.py         # 带符号 AIG 数据加载器
  utils.py             # 边划分、图上下文构造和工具函数
  __init__.py          # 包导出
```

## 环境配置

推荐使用 Conda 创建独立环境：

```bash
conda env create -f environment.yml
conda activate phaseconegnn
```

也可以手动创建环境：

```bash
conda create -n phaseconegnn python=3.10 -y
conda activate phaseconegnn
conda install pytorch numpy tqdm -c pytorch -c conda-forge -y
```

如果需要使用 CUDA 进行 GPU 训练，请安装与本机 CUDA 驱动匹配的 PyTorch 版本。例如：

```bash
conda install pytorch pytorch-cuda=12.1 numpy tqdm -c pytorch -c nvidia -c conda-forge -y
```

仅当使用 `--feature_type spectral` 时，才需要额外安装谱特征相关依赖：

```bash
conda install scipy scikit-learn -c conda-forge -y
```

## 数据集获取

`train.py` 使用的是处理后的带符号 AIG 样本。完整处理后数据集包含较大的 `.npz` 文件以及大量逐图 CSV 文件，因此单独发布在 Hugging Face Datasets：

[Noah0519/AIGdataset](https://huggingface.co/datasets/Noah0519/AIGdataset)

下载后请通过 `--data_root_path` 指向处理后数据集根目录。如果将处理后数据集放在仓库根目录下，建议使用如下相对路径：

```bash
--data_root_path ./AIGDataset/ForgeEDA_proxy_processed
```

## Demo 处理后数据集

本仓库内包含一个很小的处理后数据集，用于检查数据加载器和训练入口是否能完整跑通：

```text
data/demo_AIGDataset/ForgeEDA_proxy_demo/
```

该 demo 只包含少量小图和一个 `demo` split，仅用于 smoke test，不用于复现实验结果。可在 `PhaseConeGNN` 目录下运行：

```bash
python train.py \
  --task_type prob \
  --data_root_path ./data/demo_AIGDataset/ForgeEDA_proxy_demo \
  --split_file demo \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 32 \
  --layer_num 2 \
  --batch_size 2 \
  --device_backend cpu \
  --epochs 1 \
  --name_others demo_smoke
```

完整实验请使用 Hugging Face 上的完整处理后数据集，并在运行时通过 `--data_root_path` 指向下载后的数据集根目录。

## 处理后数据格式

数据加载器期望处理后的数据集具有如下结构：

```text
AIGDataset/ForgeEDA_proxy_processed/
  split/
    0.05-0.05-0.9__full/
      train.txt
      valid.txt
      test.txt
  npz/
    labels.npz
  <graph_name>/
    raw/
      signed_edge.csv
      node-feat.csv
      prob.csv
```

其中，`train.txt`、`valid.txt` 和 `test.txt` 可以存放相对于数据集根目录的图路径，也可以存放绝对路径。每个图目录需要包含：

- `raw/signed_edge.csv`：带符号有向边，格式为 `(src, dst, sign)`。正号表示相位保持边，负号表示反相边；
- `raw/node-feat.csv`：节点输入特征；
- `raw/prob.csv`：节点信号概率标签。

对于等效门识别和真值表距离学习任务，`npz/labels.npz` 存储 gate-pair 标签。加载器读取其中的 `tt_pair_index` 和 `tt_dis`，并使用 `eq_sim = 1 - tt_dis` 得到等效门相似性标签。

## 运行实验

以下命令均在 `PhaseConeGNN` 目录下运行。

### 信号概率预测

```bash
python train.py \
  --task_type prob \
  --data_root_path ./AIGDataset/ForgeEDA_proxy_processed \
  --split_file 0.05-0.05-0.9__full \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 256 \
  --layer_num 5 \
  --batch_size 128 \
  --device_backend cuda \
  --device 0 \
  --epochs 40 \
  --name_others forgeeda_proxy_full
```

### 等效门识别

```bash
python train.py \
  --task_type eq \
  --data_root_path ./AIGDataset/ForgeEDA_proxy_processed \
  --split_file 0.05-0.05-0.9__full \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 256 \
  --layer_num 5 \
  --batch_size 128 \
  --device_backend cuda \
  --device 0 \
  --epochs 40 \
  --name_others forgeeda_proxy_full
```

### 真值表距离学习

```bash
python train.py \
  --task_type tt \
  --data_root_path ./AIGDataset/ForgeEDA_proxy_processed \
  --split_file 0.05-0.05-0.9__full \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 256 \
  --layer_num 5 \
  --batch_size 128 \
  --device_backend cuda \
  --device 0 \
  --epochs 40 \
  --name_others forgeeda_proxy_full
```

如果只使用 CPU，将设备参数替换为：

```bash
--device_backend cpu --device -1
```

## 主要参数

- `--task_type`：训练任务，可选 `prob`、`eq` 或 `tt`。
- `--data_root_path`：处理后数据集根目录。
- `--split_file`：`split/` 下的数据划分文件夹名称。
- `--feature_type`：输入特征类型，可选 `one-hot`、`raw` 或 `spectral`。
- `--in_dim`：输入特征维度。使用 `one-hot` 时，该参数表示门类型数量。
- `--out_dim`：节点嵌入维度。由于模型同时维护原相位和反相位状态，该参数必须为偶数。
- `--layer_num`：传播层数。
- `--use_backward`：`-1` 表示使用任务相关默认设置，`0` 表示关闭反向 fanout 传播，`1` 表示开启反向 fanout 传播。
- `--use_layer_moe`：是否开启多深度表示自适应融合。
- `--lambda_not`、`--lambda_and`：信号概率预测中 NOT/AND 逻辑一致性损失的权重。

## 输出文件

训练过程中会在当前工作目录下生成：

```text
ft_saved/   # 最优模型参数
results/    # 实验日志和指标结果
```

脚本会输出所选任务的验证集和测试集指标。对于等效门识别，gate-pair 相似度由两个门节点嵌入的余弦相似度计算得到。
