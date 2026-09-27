# my_experiment.py

# 1. 从我们提供的帮助脚本中导入加载函数
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone

RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

# 2. 调用函数来获取数据
#    这个函数会自动读取 .csv 文件并返回你需要的所有内容
X_train, y_train, X_test_unlabeled = load_data()

# 3. (验证步骤) 检查一下数据是否加载成功
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 打印第一个训练样本和它的标签，感受一下数据
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 打印第一个需要你预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# --- 在这里开始你的实验！ ---
# 现在，你可以使用 X_train, y_train, 和 X_test_unlabeled 这三个变量
# 来进行后续的特征提取、模型训练和预测了。

# 从原训练集中划分 20% 作为验证集
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
print("-" * 20)

# 1. 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_split)
X_val_tfidf = vectorizer.transform(X_val)
print(f"TF-IDF 特征数量: {X_train_tfidf.shape[1]}")
print("=" * 50)

# 2. 训练一个模型...
models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Linear SVM": SVC(
        kernel='linear',
        C=1.0,
        random_state=RANDOM_STATE
    )
}

results = []

best_model_name = None
best_model = None
best_f1 = -1


for model_name, model in models.items():

    print(f"\n--- 开始训练 {model_name} ---")

    start_time = time.time()

    # 训练模型
    model.fit(X_train_tfidf, y_train_split)

    train_time = time.time() - start_time

    # 在验证集上预测
    y_pred = model.predict(X_val_tfidf)

    # 计算评价指标
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(
        y_val,
        y_pred,
        average='macro'
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"训练时间  : {train_time:.2f} 秒")

    # 保存实验结果
    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time": train_time
    })

    # 当前使用 Macro-F1 选择最佳模型
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model_name = model_name
        best_model = model


# 将模型比较结果保存下来
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False
)

print("\n" + "=" * 50)
print("--- Baseline 模型比较结果 ---")
print(results_df.to_string(index=False))
print("=" * 50)

results_df.to_csv(
    'baseline_results.csv',
    index=False
)

print(f"\n当前最佳模型: {best_model_name}")
print(f"最佳验证集 Macro-F1: {best_f1:.4f}")


# 3. 使用最佳模型重新训练，并对测试集进行预测

print(f"\n--- 使用完整训练集重新训练 {best_model_name} ---")

# 重新创建 TF-IDF 向量化器
final_vectorizer = TfidfVectorizer(
    max_features=5000
)

# 在全部有标签训练数据上重新 fit
X_train_full_tfidf = final_vectorizer.fit_transform(X_train)

# 测试集只能 transform，不能 fit
X_test_tfidf = final_vectorizer.transform(X_test_unlabeled)

# 复制已经选择好的最佳模型及其参数
final_model = clone(best_model)

# 使用全部训练数据重新训练
final_model.fit(
    X_train_full_tfidf,
    y_train
)

# 对测试集进行预测
predictions = final_model.predict(
    X_test_tfidf
)

print("测试集预测完成！")
print(f"预测结果数量: {len(predictions)}")



# 4. 保存预测结果
pd.DataFrame(predictions).to_csv(
    'predictions.csv',
    index=False,
    header=False
)

print("\npredictions.csv 已保存！")