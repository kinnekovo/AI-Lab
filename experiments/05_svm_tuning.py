# experiment5_svm_tuning.py
# Experiment 5: Linear SVM 的 C 参数调优
# 固定数据划分与 TF-IDF 设置，只改变 C。

import time
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values
    X_test_unlabeled = test_df["text"].astype(str).tolist()

    return X_train, y_train, X_test_unlabeled


# 1. 加载数据
X_train, y_train, X_test_unlabeled = load_data()

print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 50)


# 2. 与 baseline 保持完全一致的数据划分
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



# 3. 与 baseline 保持一致的 TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train_split)
X_val_tfidf = vectorizer.transform(X_val)

print(f"TF-IDF 特征数量: {X_train_tfidf.shape[1]}")
print("=" * 50)



# 4. Experiment 5：Linear SVM 的 C 参数调优
#    固定 kernel='linear'，只改变 C
C_values = [
    0.1,
    1,
    10,
    100
]

results = []

for C in C_values:

    print(f"\n--- Linear SVM: C={C} ---")

    model = SVC(
        kernel="linear",
        C=C,
        random_state=RANDOM_STATE
    )

    start_time = time.time()

    model.fit(
        X_train_tfidf,
        y_train_split
    )

    train_time = time.time() - start_time

    # 训练集表现
    y_train_pred = model.predict(
        X_train_tfidf
    )

    train_accuracy = accuracy_score(
        y_train_split,
        y_train_pred
    )

    train_macro_f1 = f1_score(
        y_train_split,
        y_train_pred,
        average="macro"
    )

    # 验证集表现
    y_val_pred = model.predict(
        X_val_tfidf
    )

    val_accuracy = accuracy_score(
        y_val,
        y_val_pred
    )

    val_macro_f1 = f1_score(
        y_val,
        y_val_pred,
        average="macro"
    )

    # 支持向量数量
    support_vector_count = len(model.support_)

    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Train Macro-F1 : {train_macro_f1:.4f}")
    print(f"Val Accuracy   : {val_accuracy:.4f}")
    print(f"Val Macro-F1   : {val_macro_f1:.4f}")
    print(f"支持向量数量   : {support_vector_count}")
    print(f"训练时间       : {train_time:.2f} 秒")

    results.append({
        "C": C,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "support_vector_count": support_vector_count,
        "train_time": train_time
    })



# 5. 汇总并保存结果
results_df = pd.DataFrame(
    results
).sort_values(
    by="val_macro_f1",
    ascending=False
)

print("\n" + "=" * 50)
print("--- Linear SVM C 参数调优结果 ---")
print(results_df.to_string(index=False))
print("=" * 50)

results_df.to_csv(
    RESULTS_DIR / "svm_tuning_results.csv",
    index=False
)

best_result = results_df.iloc[0]

print(f"\n当前最佳 C: {best_result['C']}")
print(f"最佳验证集 Accuracy: {best_result['val_accuracy']:.4f}")
print(f"最佳验证集 Macro-F1: {best_result['val_macro_f1']:.4f}")
print(f"对应训练集 Macro-F1: {best_result['train_macro_f1']:.4f}")
print(f"支持向量数量: {int(best_result['support_vector_count'])}")
print(f"训练时间: {best_result['train_time']:.2f} 秒")
print("\nsvm_tuning_results.csv 已保存！")
