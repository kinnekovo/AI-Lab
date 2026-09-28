# experiment9_final_comparison.py
# Experiment 9: 最终公平横向比较（更新版）
# 统一使用选定的 TF-IDF 配置：
#   max_features=20000
#   ngram_range=(1, 1)
# 比较 tuned MLP / Logistic Regression / Linear SVM / RBF SVM。

import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


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
# 3. 最终统一 TF-IDF 配置
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
print("=" * 50)


# ============================================================
# 4. 定义调参后的四个最终候选模型
# ============================================================

models = {
    "MLP": MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        alpha=0.001,
        max_iter=300,
        random_state=RANDOM_STATE
    ),

    "Logistic Regression": LogisticRegression(
        C=300,
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Linear SVM": SVC(
        kernel="linear",
        C=300,
        random_state=RANDOM_STATE
    ),

    "RBF SVM": SVC(
        kernel="rbf",
        C=100,
        gamma="scale",
        random_state=RANDOM_STATE
    )
}


# ============================================================
# 5. 公平比较
# ============================================================

results = []

for model_name, model in models.items():

    print(f"\n--- 开始训练 {model_name} ---")

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

    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Train Macro-F1 : {train_macro_f1:.4f}")
    print(f"Val Accuracy   : {val_accuracy:.4f}")
    print(f"Val Macro-F1   : {val_macro_f1:.4f}")
    print(f"训练时间       : {train_time:.2f} 秒")

    extra_info = ""

    if model_name == "MLP":
        extra_info = f"n_iter={model.n_iter_}"
        print(f"实际迭代次数   : {model.n_iter_}")

    elif model_name == "Logistic Regression":
        actual_n_iter = int(model.n_iter_.max())
        extra_info = f"n_iter={actual_n_iter}"
        print(f"实际迭代次数   : {actual_n_iter}")

    elif model_name in ["Linear SVM", "RBF SVM"]:
        support_vector_count = len(model.support_)
        extra_info = f"support_vectors={support_vector_count}"
        print(f"支持向量数量   : {support_vector_count}")

    results.append({
        "model": model_name,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "train_time": train_time,
        "extra_info": extra_info
    })


# ============================================================
# 6. 汇总并保存最终比较结果
# ============================================================

results_df = pd.DataFrame(
    results
).sort_values(
    by="val_macro_f1",
    ascending=False
)

print("\n" + "=" * 50)
print("--- 最终公平比较结果（含 RBF SVM） ---")
print(results_df.to_string(index=False))
print("=" * 50)

results_df.to_csv(
    "final_model_comparison_results.csv",
    index=False
)

best_result = results_df.iloc[0]

print(f"\n当前最佳模型: {best_result['model']}")
print(f"最佳验证集 Accuracy: {best_result['val_accuracy']:.4f}")
print(f"最佳验证集 Macro-F1: {best_result['val_macro_f1']:.4f}")
print(f"对应训练集 Macro-F1: {best_result['train_macro_f1']:.4f}")
print(f"训练时间: {best_result['train_time']:.2f} 秒")

print("\nfinal_model_comparison_results.csv 已保存！")
