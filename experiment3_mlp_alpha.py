# experiment3_mlp_alpha.py
# Experiment 3: MLP 正则化参数 alpha 调参
# 固定数据划分、TF-IDF 和网络结构，只改变 alpha。

import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')

    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()

    return X_train, y_train, X_test_unlabeled



# 1. 加载数据
X_train, y_train, X_test_unlabeled = load_data()
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 50)


# 2. 划分训练集和验证集
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


# 3. TF-IDF 特征提取
vectorizer = TfidfVectorizer(
    max_features=5000
)

# 只在训练子集上 fit，避免验证集信息泄露
X_train_tfidf = vectorizer.fit_transform(X_train_split)
X_val_tfidf = vectorizer.transform(X_val)

print(f"TF-IDF 特征数量: {X_train_tfidf.shape[1]}")
print("=" * 50)


# 4. Experiment 3：MLP alpha 调参

alpha_values = [
    0.0001,   # baseline
    0.001,
    0.01,
    0.1
]

results = []

for alpha in alpha_values:

    print(f"\n--- MLP: alpha={alpha} ---")

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        alpha=alpha,
        max_iter=300,
        random_state=RANDOM_STATE
    )

    start_time = time.time()

    model.fit(
        X_train_tfidf,
        y_train_split
    )

    train_time = time.time() - start_time

    # 训练集表现：用于观察正则化是否抑制过拟合
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
        average='macro'
    )

    # 验证集表现：作为 alpha 选择依据
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
        average='macro'
    )


    # 输出当前实验结果
    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Train Macro-F1 : {train_macro_f1:.4f}")
    print(f"Val Accuracy   : {val_accuracy:.4f}")
    print(f"Val Macro-F1   : {val_macro_f1:.4f}")
    print(f"实际迭代次数   : {model.n_iter_}")
    print(f"训练时间       : {train_time:.2f} 秒")


    # 保存结果
    results.append({
        "alpha": alpha,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "n_iter": model.n_iter_,
        "train_time": train_time
    })



# 5. 汇总并保存实验结果
results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="val_macro_f1",
    ascending=False
)

print("\n" + "=" * 50)
print("--- MLP alpha 调参结果 ---")
print(
    results_df.to_string(
        index=False
    )
)
print("=" * 50)

results_df.to_csv(
    'mlp_alpha_results.csv',
    index=False
)


# 6. 输出当前最佳 alpha   
best_result = results_df.iloc[0]

print(
    f"\n当前最佳 alpha: "
    f"{best_result['alpha']}"
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
    f"训练时间: "
    f"{best_result['train_time']:.2f} 秒"
)

print("\nmlp_alpha_results.csv 已保存！")
