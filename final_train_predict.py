# final_train_predict.py
# 最终全量训练 + 测试集预测
#
# 使用已经通过验证集实验确定的最终配置：
# TF-IDF:
#   max_features=20000
#   ngram_range=(1, 1)
#
# MLP:
#   hidden_layer_sizes=(100,)
#   activation='relu'
#   alpha=0.001
#   max_iter=300
#   random_state=42
#
# 注意：
# 1. 此文件不再划分验证集。
# 2. 使用全部 7368 条有标签训练样本重新 fit TF-IDF 和 MLP。
# 3. 测试集只做 transform 和 predict。
# 4. 最终生成 predictions.csv。

import time
from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values
    X_test_unlabeled = test_df["text"].astype(str).tolist()

    return X_train, y_train, X_test_unlabeled


# 1. 加载全部训练数据与无标签测试数据
X_train, y_train, X_test_unlabeled = load_data()

print("--- 数据加载成功 ---")
print(f"完整训练集样本数量: {len(X_train)}")
print(f"完整训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 50)


# 2. 使用最终选定的 TF-IDF 配置
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 1)
)

print("--- 开始构建最终 TF-IDF 特征 ---")

feature_start = time.time()

# 最终阶段可以在全部有标签训练数据上 fit
X_train_tfidf = vectorizer.fit_transform(
    X_train
)

# 测试集只 transform，不参与 fit
X_test_tfidf = vectorizer.transform(
    X_test_unlabeled
)

feature_time = time.time() - feature_start

print(f"实际特征数量: {X_train_tfidf.shape[1]}")
print(f"TF-IDF 特征构建耗时: {feature_time:.2f} 秒")
print("-" * 50)


# 3. 使用最终选定的 MLP 配置
final_model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    alpha=0.001,
    max_iter=300,
    random_state=RANDOM_STATE
)

print("--- 开始使用全部训练数据训练最终 MLP ---")

train_start = time.time()

final_model.fit(
    X_train_tfidf,
    y_train
)

train_time = time.time() - train_start

print("最终模型训练完成！")
print(f"实际迭代次数: {final_model.n_iter_}")
print(f"训练时间: {train_time:.2f} 秒")
print("-" * 50)


# 4. 对无标签测试集进行预测
print("--- 开始预测测试集 ---")

predict_start = time.time()

predictions = final_model.predict(
    X_test_tfidf
)

predict_time = time.time() - predict_start

print("测试集预测完成！")
print(f"预测结果数量: {len(predictions)}")
print(f"预测耗时: {predict_time:.2f} 秒")
print("-" * 50)


# 5. 保存最终 predictions.csv
pd.DataFrame(
    predictions
).to_csv(
    OUTPUTS_DIR / "predictions.csv",
    index=False,
    header=False
)

print("predictions.csv 已保存！")


# 6. 保存一份最终配置记录，方便 README / 报告复现
config_df = pd.DataFrame([{
    "random_state": RANDOM_STATE,
    "tfidf_max_features": 20000,
    "tfidf_ngram_range": "(1, 1)",
    "model": "MLPClassifier",
    "hidden_layer_sizes": "(100,)",
    "activation": "relu",
    "alpha": 0.001,
    "max_iter": 300,
    "actual_n_iter": final_model.n_iter_,
    "train_samples": len(X_train),
    "test_samples": len(X_test_unlabeled),
    "feature_count": X_train_tfidf.shape[1],
    "feature_time": feature_time,
    "train_time": train_time,
    "predict_time": predict_time
}])

config_df.to_csv(
    OUTPUTS_DIR / "final_training_config.csv",
    index=False
)

print("final_training_config.csv 已保存！")
