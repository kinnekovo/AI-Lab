# experiment4_tfidf_features.py
# Experiment 4: TF-IDF 特征表示优化
# 固定当前最佳 MLP 配置，只研究 TF-IDF 的 max_features 与 ngram_range。

import time
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

MLP_CONFIG = {
    "hidden_layer_sizes": (100,),
    "activation": "relu",
    "alpha": 0.001,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values
    X_test_unlabeled = test_df["text"].astype(str).tolist()

    return X_train, y_train, X_test_unlabeled


def evaluate_tfidf_setting(
    X_train_split,
    X_val,
    y_train_split,
    y_val,
    max_features,
    ngram_range,
):
    print(
        f"\n--- TF-IDF: max_features={max_features}, "
        f"ngram_range={ngram_range} ---"
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    feature_start = time.time()

    X_train_tfidf = vectorizer.fit_transform(X_train_split)
    X_val_tfidf = vectorizer.transform(X_val)

    feature_time = time.time() - feature_start
    actual_features = X_train_tfidf.shape[1]

    model = MLPClassifier(**MLP_CONFIG)

    train_start = time.time()
    model.fit(X_train_tfidf, y_train_split)
    train_time = time.time() - train_start

    y_train_pred = model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(y_train_split, y_train_pred)
    train_macro_f1 = f1_score(
        y_train_split, y_train_pred, average="macro"
    )

    y_val_pred = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_macro_f1 = f1_score(
        y_val, y_val_pred, average="macro"
    )

    print(f"实际特征数量    : {actual_features}")
    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Train Macro-F1 : {train_macro_f1:.4f}")
    print(f"Val Accuracy   : {val_accuracy:.4f}")
    print(f"Val Macro-F1   : {val_macro_f1:.4f}")
    print(f"实际迭代次数   : {model.n_iter_}")
    print(f"特征提取时间   : {feature_time:.2f} 秒")
    print(f"模型训练时间   : {train_time:.2f} 秒")

    return {
        "max_features": max_features,
        "ngram_range": str(ngram_range),
        "actual_features": actual_features,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "n_iter": model.n_iter_,
        "feature_time": feature_time,
        "train_time": train_time,
    }


# 1. 加载数据
X_train, y_train, X_test_unlabeled = load_data()

print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 50)


# 2. 与前面实验保持完全一致的数据划分
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


# 3. Experiment 4A：研究 max_features
print("\n" + "=" * 50)
print("--- Experiment 4A: max_features 对比 ---")

max_features_values = [5000, 10000, 20000]
max_features_results = []

for max_features in max_features_values:
    result = evaluate_tfidf_setting(
        X_train_split,
        X_val,
        y_train_split,
        y_val,
        max_features,
        (1, 1),
    )
    max_features_results.append(result)

max_features_df = pd.DataFrame(
    max_features_results
).sort_values(
    by="val_macro_f1",
    ascending=False
)

print("\n" + "=" * 50)
print("--- max_features 实验结果 ---")
print(max_features_df.to_string(index=False))
print("=" * 50)

max_features_df.to_csv(
    RESULTS_DIR / "tfidf_max_features_results.csv",
    index=False
)

best_max_features = int(
    max_features_df.iloc[0]["max_features"]
)

print(f"\n当前最佳 max_features: {best_max_features}")
print(
    f"对应最佳验证集 Macro-F1: "
    f"{max_features_df.iloc[0]['val_macro_f1']:.4f}"
)


# 4. Experiment 4B：研究 ngram_range
print("\n" + "=" * 50)
print("--- Experiment 4B: ngram_range 对比 ---")

ngram_ranges = [(1, 1), (1, 2)]
ngram_results = []

for ngram_range in ngram_ranges:
    result = evaluate_tfidf_setting(
        X_train_split,
        X_val,
        y_train_split,
        y_val,
        best_max_features,
        ngram_range,
    )
    ngram_results.append(result)

ngram_df = pd.DataFrame(
    ngram_results
).sort_values(
    by="val_macro_f1",
    ascending=False
)

print("\n" + "=" * 50)
print("--- ngram_range 实验结果 ---")
print(ngram_df.to_string(index=False))
print("=" * 50)

ngram_df.to_csv(
    RESULTS_DIR / "tfidf_ngram_results.csv",
    index=False
)

best_ngram_result = ngram_df.iloc[0]

print(
    f"\n当前最佳 ngram_range: "
    f"{best_ngram_result['ngram_range']}"
)
print(
    f"最佳验证集 Accuracy: "
    f"{best_ngram_result['val_accuracy']:.4f}"
)
print(
    f"最佳验证集 Macro-F1: "
    f"{best_ngram_result['val_macro_f1']:.4f}"
)

print(
    "\n结果文件已保存："
    "\n1. tfidf_max_features_results.csv"
    "\n2. tfidf_ngram_results.csv"
)
