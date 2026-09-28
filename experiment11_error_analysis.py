# experiment11_error_analysis.py
# Experiment 11: 最终模型错误分析
# 使用最终选定的 TF-IDF + MLP 配置，在固定验证集上进行：
# 1. overall metrics n
# 2. per-class precision / recall / F1
# 3. confusion matrix
# 4. 最常见混淆类别对
# 5. 典型误分类样本导出
#
# 注意：本文件只用于验证集分析，不生成最终 test 预测。

import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (   
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data_unlabeled.csv")

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values
    X_test_unlabeled = test_df["text"].astype(str).tolist()

    return X_train, y_train, X_test_unlabeled


# ============================================================
# 1. 加载数据
# ============================================================

X_train, y_train, X_test_unlabeled = load_data()

print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 50)


# ============================================================
# 2. 与前面实验保持完全一致的数据划分
# ============================================================

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("--- 数据集划分完成 ---")
print(f"实际训练集样本数量: {len(X_train_split)}")
print(f"验证集样本数量: {len(X_val)}")
print("-" * 50)


# ============================================================
# 3. 最终选定的 TF-IDF 配置
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 1)
)

feature_start = time.time()

X_train_tfidf = vectorizer.fit_transform(X_train_split)
X_val_tfidf = vectorizer.transform(X_val)

feature_time = time.time() - feature_start

print("--- TF-IDF 特征提取完成 ---")
print(f"实际特征数量: {X_train_tfidf.shape[1]}")
print(f"特征提取时间: {feature_time:.2f} 秒")
print("-" * 50)


# ============================================================
# 4. 最终选定的 MLP 配置
# ============================================================

model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    alpha=0.001,
    max_iter=300,
    random_state=RANDOM_STATE
)

print("--- 开始训练最终 MLP ---")

start_time = time.time()

model.fit(
    X_train_tfidf,
    y_train_split
)

train_time = time.time() - start_time

print(f"训练完成，耗时: {train_time:.2f} 秒")
print(f"实际迭代次数: {model.n_iter_}")


# ============================================================
# 5. 验证集总体表现
# ============================================================

y_val_pred = model.predict(X_val_tfidf)

val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

val_macro_f1 = f1_score(
    y_val,
    y_val_pred,
    average="macro"
)

print("\n" + "=" * 50)
print("--- 最终模型验证集总体结果 ---")
print(f"Validation Accuracy : {val_accuracy:.4f}")
print(f"Validation Macro-F1 : {val_macro_f1:.4f}")
print("=" * 50)


# ============================================================
# 6. Per-class classification report
# ============================================================

report_dict = classification_report(
    y_val,
    y_val_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_df.to_csv(
    "final_classification_report.csv"
)

print("\n--- Per-class Classification Report ---")
print(report_df.to_string())


# ============================================================
# 7. Confusion Matrix
# ============================================================

labels = sorted(set(y_val))

cm = confusion_matrix(
    y_val,
    y_val_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{label}" for label in labels],
    columns=[f"pred_{label}" for label in labels]
)

cm_df.to_csv(
    "final_confusion_matrix.csv"
)

print("\n--- Confusion Matrix ---")
print(cm_df.to_string())


# ============================================================
# 8. 保存 Confusion Matrix 图
# ============================================================

plt.figure(figsize=(8, 7))
plt.imshow(cm)
plt.title("Confusion Matrix - Final MLP")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(
    "final_confusion_matrix.png",
    dpi=200
)
plt.close()

print("\nfinal_confusion_matrix.png 已保存！")


# ============================================================
# 9. 找出最常见的混淆类别对
# ============================================================

confusion_pairs = []

for i, true_label in enumerate(labels):
    for j, pred_label in enumerate(labels):

        if i == j:
            continue

        count = cm[i, j]

        if count > 0:
            confusion_pairs.append({
                "true_label": true_label,
                "predicted_label": pred_label,
                "count": int(count)
            })

confusion_pairs_df = pd.DataFrame(
    confusion_pairs
).sort_values(
    by="count",
    ascending=False
)

confusion_pairs_df.to_csv(
    "top_confusion_pairs.csv",
    index=False
)

print("\n--- 最常见混淆类别对 Top 10 ---")
print(
    confusion_pairs_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 10. 导出典型误分类样本
# ============================================================

error_records = []

for text, true_label, pred_label in zip(
    X_val,
    y_val,
    y_val_pred
):

    if true_label != pred_label:
        error_records.append({
            "true_label": true_label,
            "predicted_label": pred_label,
            "text": text
        })

errors_df = pd.DataFrame(
    error_records
)

errors_df.to_csv(
    "misclassified_samples.csv",
    index=False
)

print(
    f"\n误分类样本数量: "
    f"{len(errors_df)}"
)

print(
    "misclassified_samples.csv 已保存！"
)


# ============================================================
# 11. 为每个主要混淆类别对导出少量样本
# ============================================================

top_pairs = confusion_pairs_df.head(5)

example_records = []

for _, row in top_pairs.iterrows():

    true_label = row["true_label"]
    pred_label = row["predicted_label"]

    pair_examples = errors_df[
        (errors_df["true_label"] == true_label) &
        (errors_df["predicted_label"] == pred_label)
    ].head(3)

    for _, example in pair_examples.iterrows():
        example_records.append({
            "true_label": true_label,
            "predicted_label": pred_label,
            "text": example["text"]
        })

examples_df = pd.DataFrame(
    example_records
)

examples_df.to_csv(
    "top_confusion_examples.csv",
    index=False
)

print(
    "top_confusion_examples.csv 已保存！"
)

print(
    "\n最终生成文件："
    "\n1. final_classification_report.csv"
    "\n2. final_confusion_matrix.csv"
    "\n3. final_confusion_matrix.png"
    "\n4. top_confusion_pairs.csv"
    "\n5. misclassified_samples.csv"
    "\n6. top_confusion_examples.csv"
)
