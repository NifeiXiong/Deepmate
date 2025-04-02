import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import lightgbm as lgb


def ML_concert(excel_address=r"F:\mating_model\mating_model\all(5-14).xlsx", model_name="KNN"):
    # 定义函数计算 PR 曲线数据并存储
    global PR_AUC
    global F1_Score

    def calculate_pr_curve(model, model_name):
        model.fit(X_train, y_train)  # 训练模型
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取正类概率

        # 计算 Precision, Recall 和 AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        # 计算 F1 分数
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        return precision, recall, pr_auc, f1

    # 定义并训练模型函数
    def train_knn():
        model = KNeighborsClassifier(n_neighbors=5)
        return calculate_pr_curve(model, 'KNN')

    def train_svm():
        model = SVC(kernel='linear', probability=True, random_state=1)
        return calculate_pr_curve(model, 'SVM')

    def train_rf():
        model = RandomForestClassifier(n_estimators=100, random_state=1)
        return calculate_pr_curve(model, 'Random Forest')

    def train_catboost():
        model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=1)
        return calculate_pr_curve(model, 'CatBoost')

    def train_dnn():
        # One-hot 编码标签
        y_train_onehot = to_categorical(y_train)
        y_test_onehot = to_categorical(y_test)

        # 定义 DNN 模型
        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 训练模型
        model.fit(X_train, y_train_onehot, epochs=50, batch_size=32, verbose=0)

        # 获取预测概率
        y_pred_proba = model.predict(X_test)[:, 1]

        # 计算 Precision, Recall 和 AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        # 获取预测类别并计算 F1 分数
        y_pred = model.predict(X_test).argmax(axis=1)
        f1 = f1_score(y_test, y_pred)

        return precision, recall, pr_auc, f1

    def train_lstm():
        # 重新 reshape 数据，LSTM 需要 3D 输入形状：样本数, 时间步数, 特征数
        X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # 定义 LSTM 模型
        model = Sequential([
            LSTM(128, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 训练 LSTM 模型
        model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)

        # 获取预测概率
        y_pred_proba = model.predict(X_test_lstm)

        # 计算 Precision, Recall 和 AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        # 获取预测类别并计算 F1 分数
        y_pred = (y_pred_proba > 0.5).astype(int)
        f1 = f1_score(y_test, y_pred)

        return precision, recall, pr_auc, f1

    def train_lightgbm():
        model = lgb.LGBMClassifier(objective='binary', metric='binary_error', random_state=1)
        return calculate_pr_curve(model, 'LightGBM')

    # 加载数据
    data = pd.read_excel(excel_address)

    # 分离特征和标签
    X = data.drop(['y'], axis=1)  # 假设列 'n' 表示扩充数量
    y = data['y']

    # 删除全是 NaN 的列
    X = X.dropna(axis=1, how='all')

    # 处理缺失值（填充为均值）
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 用户输入模型名称并运行对应的模型
    model_dict = {
        'KNN': train_knn,
        'SVM': train_svm,
        'Random Forest': train_rf,
        'CatBoost': train_catboost,
        'DNN': train_dnn,
        'LSTM': train_lstm,
        'LightGBM': train_lightgbm
    }

    # 获取用户输入的模型名称
    # "请输入模型名称（KNN, SVM, Random Forest, CatBoost, DNN, LSTM, LightGBM）："
    model_name = model_name

    # 确保用户输入的模型名称有效
    if model_name in model_dict:
        precision, recall, pr_auc, f1 = model_dict[model_name]()  # 调用对应模型的训练函数
        PR_AUC = pr_auc
        F1_Score = f1
        print(f'{model_name} - PR AUC: {pr_auc:.4f} - F1 Score: {f1:.4f}')

        # 绘制 PR 曲线
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.4f}, F1 = {f1:.4f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(f"{model_name}.png", dpi=120)
        plt.show()
    else:
        print("输入的模型名称无效，请输入有效的模型名称。")

    return PR_AUC, F1_Score

if __name__ == '__main__':
    excel_address = r"F:\mating_model\mating_model\all(5-14).xlsx"
    model_name="KNN"
    ML_concert(excel_address, model_name)
