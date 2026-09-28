# experiment8_tfidf_boundary.py
# Experiment 8: TF-IDF max_features 边界补充实验
# 仅新增 max_features=40000；其他设置与前面实验保持一致。

import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42

def load_data():
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data_unlabeled.csv")
    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values
    X_test_unlabeled = test_df["text"].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

X_train, y_train, X_test_unlabeled = load_data()

print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 50)

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

MAX_FEATURES = 40000
NGRAM_RANGE = (1, 1)

print("\n" + "=" * 50)
print("--- Experiment 8: TF-IDF 边界补充实验 ---")
print(f"max_features = {MAX_FEATURES}")
print(f"ngram_range  = {NGRAM_RANGE}")

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE
)

feature_start = time.time()
X_train_tfidf = vectorizer.fit_transform(X_train_split)
X_val_tfidf = vectorizer.transform(X_val)
feature_time = time.time() - feature_start

actual_features = X_train_tfidf.shape[1]

model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    alpha=0.001,
    max_iter=300,
    random_state=RANDOM_STATE
)

train_start = time.time()
model.fit(X_train_tfidf, y_train_split)
train_time = time.time() - train_start

y_train_pred = model.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train_split, y_train_pred)
train_macro_f1 = f1_score(y_train_split, y_train_pred, average="macro")

y_val_pred = model.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")

print(f"实际特征数量    : {actual_features}")
print(f"Train Accuracy : {train_accuracy:.4f}")
print(f"Train Macro-F1 : {train_macro_f1:.4f}")
print(f"Val Accuracy   : {val_accuracy:.4f}")
print(f"Val Macro-F1   : {val_macro_f1:.4f}")
print(f"实际迭代次数   : {model.n_iter_}")
print(f"特征提取时间   : {feature_time:.2f} 秒")
print(f"模型训练时间   : {train_time:.2f} 秒")

REFERENCE_20000_VAL_ACCURACY = 0.933514
REFERENCE_20000_VAL_MACRO_F1 = 0.934024

accuracy_change = val_accuracy - REFERENCE_20000_VAL_ACCURACY
f1_change = val_macro_f1 - REFERENCE_20000_VAL_MACRO_F1

print("\n" + "=" * 50)
print("--- 与 20000 特征结果比较 ---")
print(f"20000 features Val Accuracy : {REFERENCE_20000_VAL_ACCURACY:.4f}")
print(f"40000 features Val Accuracy : {val_accuracy:.4f}")
print(f"Accuracy 变化               : {accuracy_change:+.4f}")
print(f"20000 features Val Macro-F1 : {REFERENCE_20000_VAL_MACRO_F1:.4f}")
print(f"40000 features Val Macro-F1 : {val_macro_f1:.4f}")
print(f"Macro-F1 变化               : {f1_change:+.4f}")
print("=" * 50)

result_df = pd.DataFrame([{
    "max_features": MAX_FEATURES,
    "ngram_range": str(NGRAM_RANGE),
    "actual_features": actual_features,
    "train_accuracy": train_accuracy,
    "train_macro_f1": train_macro_f1,
    "val_accuracy": val_accuracy,
    "val_macro_f1": val_macro_f1,
    "n_iter": model.n_iter_,
    "feature_time": feature_time,
    "train_time": train_time,
    "reference_20000_val_accuracy": REFERENCE_20000_VAL_ACCURACY,
    "reference_20000_val_macro_f1": REFERENCE_20000_VAL_MACRO_F1,
    "accuracy_change_vs_20000": accuracy_change,
    "macro_f1_change_vs_20000": f1_change
}])

result_df.to_csv("tfidf_boundary_40000_results.csv", index=False)

print("\ntfidf_boundary_40000_results.csv 已保存！")
