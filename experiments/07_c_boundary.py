# experiment7_c_boundary.py
# Experiment 7: SVM / Logistic Regression 的 C 边界补充实验
# 在前面已测试 C=[0.1, 1, 10, 100] 的基础上，
# 继续测试 C=300 和 C=1000，判断性能是否仍在提升。

import time
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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


def evaluate_model(model_name, model, X_train_tfidf, X_val_tfidf,
                   y_train_split, y_val, C):
    print(f"\n--- {model_name}: C={C} ---")

    start_time = time.time()
    model.fit(X_train_tfidf, y_train_split)
    train_time = time.time() - start_time

    # 训练集表现
    y_train_pred = model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(y_train_split, y_train_pred)
    train_macro_f1 = f1_score(
        y_train_split,
        y_train_pred,
        average="macro"
    )

    # 验证集表现
    y_val_pred = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
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

    result = {
        "model": model_name,
        "C": C,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "train_time": train_time
    }

    # 保存模型特有信息
    if model_name == "Linear SVM":
        result["support_vector_count"] = len(model.support_)
        result["n_iter"] = None
        print(f"支持向量数量   : {len(model.support_)}")

    elif model_name == "Logistic Regression":
        result["support_vector_count"] = None
        result["n_iter"] = int(model.n_iter_.max())
        print(f"实际迭代次数   : {int(model.n_iter_.max())}")

    return result


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
# 3. 与前面实验保持一致的 TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train_split)
X_val_tfidf = vectorizer.transform(X_val)

print(f"TF-IDF 特征数量: {X_train_tfidf.shape[1]}")
print("=" * 50)


# ============================================================
# 4. 边界补充实验
#    C=100 是前面搜索范围中的最佳值，这里作为参考点保留，
#    再测试 300 和 1000 判断是否仍有提升。
# ============================================================

C_values = [
    100,
    300,
    1000
]

results = []


# -------------------------
# Logistic Regression
# -------------------------

print("\n" + "=" * 50)
print("--- Logistic Regression: C 边界实验 ---")

for C in C_values:

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    result = evaluate_model(
        model_name="Logistic Regression",
        model=model,
        X_train_tfidf=X_train_tfidf,
        X_val_tfidf=X_val_tfidf,
        y_train_split=y_train_split,
        y_val=y_val,
        C=C
    )

    results.append(result)


# -------------------------
# Linear SVM
# -------------------------

print("\n" + "=" * 50)
print("--- Linear SVM: C 边界实验 ---")

for C in C_values:

    model = SVC(
        kernel="linear",
        C=C,
        random_state=RANDOM_STATE
    )

    result = evaluate_model(
        model_name="Linear SVM",
        model=model,
        X_train_tfidf=X_train_tfidf,
        X_val_tfidf=X_val_tfidf,
        y_train_split=y_train_split,
        y_val=y_val,
        C=C
    )

    results.append(result)


# ============================================================
# 5. 汇总并保存结果
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("--- C 边界补充实验结果 ---")

print(
    results_df.sort_values(
        by=["model", "val_macro_f1"],
        ascending=[True, False]
    ).to_string(index=False)
)

print("=" * 50)

results_df.to_csv(
    RESULTS_DIR / "c_boundary_results.csv",
    index=False
)


# ============================================================
# 6. 分模型输出最佳结果
# ============================================================

for model_name in [
    "Logistic Regression",
    "Linear SVM"
]:

    model_results = results_df[
        results_df["model"] == model_name
    ].sort_values(
        by="val_macro_f1",
        ascending=False
    )

    best_result = model_results.iloc[0]

    print(
        f"\n{model_name} 在本轮最佳 C: "
        f"{best_result['C']}"
    )

    print(
        f"最佳验证集 Macro-F1: "
        f"{best_result['val_macro_f1']:.4f}"
    )

print("\nc_boundary_results.csv 已保存！")
