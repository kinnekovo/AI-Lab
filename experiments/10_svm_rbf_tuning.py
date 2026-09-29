# experiment10_svm_rbf_tuning.py
# Experiment 10: RBF SVM 调参
# Phase A：固定 gamma='scale'，搜索 C
# Phase B：固定 Phase A 最佳 C，搜索 gamma
# 为保证与此前 Linear SVM 调参公平比较，统一使用：
#   TF-IDF max_features=5000
#   ngram_range=(1, 1)
#   相同 train/validation split

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


def evaluate_rbf_svm(
    C,
    gamma,
    X_train_tfidf,
    X_val_tfidf,
    y_train_split,
    y_val,
):
    print(f"\n--- RBF SVM: C={C}, gamma={gamma} ---")

    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
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

    support_vector_count = len(model.support_)

    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Train Macro-F1 : {train_macro_f1:.4f}")
    print(f"Val Accuracy   : {val_accuracy:.4f}")
    print(f"Val Macro-F1   : {val_macro_f1:.4f}")
    print(f"支持向量数量   : {support_vector_count}")
    print(f"训练时间       : {train_time:.2f} 秒")

    return {
        "C": C,
        "gamma": str(gamma),
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "support_vector_count": support_vector_count,
        "train_time": train_time
    }


# 1. 加载数据
X_train, y_train, X_test_unlabeled = load_data()

print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 50)


# 2. 与此前实验保持完全一致的数据划分
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


# 3. 与此前 SVM 调参保持一致的 TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 1)
)

X_train_tfidf = vectorizer.fit_transform(
    X_train_split
)

X_val_tfidf = vectorizer.transform(
    X_val
)

print(f"TF-IDF 特征数量: {X_train_tfidf.shape[1]}")
print("=" * 50)


# 4. Phase A：固定 gamma='scale'，搜索 C
print("\n" + "=" * 50)
print("--- Phase A: RBF SVM 的 C 参数搜索 ---")

C_values = [
    1,
    10,
    100,
    300
]

phase_a_results = []

for C in C_values:

    result = evaluate_rbf_svm(
        C=C,
        gamma="scale",
        X_train_tfidf=X_train_tfidf,
        X_val_tfidf=X_val_tfidf,
        y_train_split=y_train_split,
        y_val=y_val
    )

    phase_a_results.append(
        result
    )


phase_a_df = pd.DataFrame(
    phase_a_results
).sort_values(
    by="val_macro_f1",
    ascending=False
)

print("\n" + "=" * 50)
print("--- Phase A 结果：固定 gamma='scale' 搜索 C ---")
print(
    phase_a_df.to_string(
        index=False
    )
)
print("=" * 50)

phase_a_df.to_csv(
    RESULTS_DIR / "svm_rbf_c_results.csv",
    index=False
)

best_C = float(
    phase_a_df.iloc[0]["C"]
)

print(
    f"\nPhase A 最佳 C: "
    f"{best_C}"
)

print(
    f"Phase A 最佳验证集 Macro-F1: "
    f"{phase_a_df.iloc[0]['val_macro_f1']:.4f}"
)


# 5. Phase B：固定最佳 C，搜索 gamma
print("\n" + "=" * 50)
print("--- Phase B: RBF SVM 的 gamma 参数搜索 ---")
print(f"固定 C = {best_C}")

gamma_values = [
    "scale",
    "auto",
    0.1,
    1
]

phase_b_results = []

for gamma in gamma_values:

    # gamma='scale' 且 C=best_C 已在 Phase A 中跑过
    # 直接复用，避免重复训练
    if gamma == "scale":

        reused = phase_a_df[
            phase_a_df["C"] == best_C
        ].iloc[0].to_dict()

        print(
            f"\n--- RBF SVM: C={best_C}, "
            f"gamma=scale（复用 Phase A 结果） ---"
        )

        print(
            f"Val Accuracy   : "
            f"{reused['val_accuracy']:.4f}"
        )

        print(
            f"Val Macro-F1   : "
            f"{reused['val_macro_f1']:.4f}"
        )

        phase_b_results.append(
            reused
        )

        continue

    result = evaluate_rbf_svm(
        C=best_C,
        gamma=gamma,
        X_train_tfidf=X_train_tfidf,
        X_val_tfidf=X_val_tfidf,
        y_train_split=y_train_split,
        y_val=y_val
    )

    phase_b_results.append(
        result
    )


phase_b_df = pd.DataFrame(
    phase_b_results
).sort_values(
    by="val_macro_f1",
    ascending=False
)

print("\n" + "=" * 50)
print("--- Phase B 结果：固定最佳 C 搜索 gamma ---")
print(
    phase_b_df.to_string(
        index=False
    )
)
print("=" * 50)

phase_b_df.to_csv(
    RESULTS_DIR / "svm_rbf_gamma_results.csv",
    index=False
)


# 6. 输出 RBF SVM 当前最佳配置
best_result = phase_b_df.iloc[0]

print(
    f"\nRBF SVM 当前最佳配置:"
)

print(
    f"C = {best_result['C']}"
)

print(
    f"gamma = {best_result['gamma']}"
)

print(
    f"最佳验证集 Accuracy: "
    f"{best_result['val_accuracy']:.4f}"
)

print(
    f"最佳验证集 Macro-F1: "
    f"{best_result['val_macro_f1']:.4f}"
)

print(
    f"对应训练集 Macro-F1: "
    f"{best_result['train_macro_f1']:.4f}"
)

print(
    f"支持向量数量: "
    f"{int(best_result['support_vector_count'])}"
)

print(
    f"训练时间: "
    f"{best_result['train_time']:.2f} 秒"
)

print(
    "\n结果文件已保存："
    "\n1. svm_rbf_c_results.csv"
    "\n2. svm_rbf_gamma_results.csv"
)
