# 实验一：文本分类

## 1. 实验任务

使用 TF-IDF 特征与传统机器学习模型 / MLP，对新闻文本进行 10 分类。

## 2. 数据集

- 有标签训练样本：7,368 条
- 无标签测试样本：2,457 条
- 验证集划分：分层 8:2 划分，训练子集 5,894 条、验证集 1,474 条
- 随机种子：`random_state=42`

## 3. 项目结构

- `data/`：原始有标签训练集与无标签测试集。
- `experiments/`：按实验顺序编号、可独立运行的实验脚本。
- `results/`：调参和模型比较的 CSV 结果；错误分析 CSV 位于 `results/error_analysis/`。
- `outputs/`：最终提交产物，包括预测结果、最终训练配置和混淆矩阵图片。
- `docs/`：调参记录和实验报告等文档。
- `final_train_predict.py`：使用全部有标签数据训练已验证的最终 MLP，并生成测试集预测结果。

## 4. 实验流程

Baseline → 模型调参 → TF-IDF 优化 → 最终公平比较 → 错误分析 → 全量数据最终训练。

## 5. 最终配置

TF-IDF：

- `max_features=20000`
- `ngram_range=(1, 1)`

MLP：

- `hidden_layer_sizes=(100,)`
- `activation='relu'`
- `alpha=0.001`
- `max_iter=300`
- `random_state=42`

验证集结果：

- Accuracy = 0.9335
- Macro-F1 = 0.9340

## 6. 生成最终预测

在项目根目录执行：

```bash
python final_train_predict.py
```

脚本会生成 `outputs/predictions.csv` 和 `outputs/final_training_config.csv`。

## 7. 复现实验

在项目根目录执行，例如：

```bash
python experiments/01_baseline.py
python experiments/09_final_comparison.py
python experiments/11_error_analysis.py
```

部分 MLP 实验运行时间较长。所有脚本均基于自身文件位置解析路径，不依赖终端启动时的当前工作目录。

## 8. 主要结果

| 模型 | Macro-F1 |
| --- | ---: |
| MLP | 0.9340 |
| Logistic Regression | 0.9311 |
| RBF SVM | 0.9268 |
| Linear SVM | 0.9239 |
