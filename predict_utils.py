import numpy as np
from train_utils import sigmoid
import matplotlib.pyplot as plt

# 线性回归模型的预测函数
def predict(X, params):
    """
    :param X: 测试集数据
    :param params: 模型训练参数
    :return: 模型预测结果
    """
    w = params["w"]
    b = params["b"]

    y_pred = np.dot(X, w) + b
    return y_pred
# 逻辑回归模型的预测函数
def predict_logit(X, params):
    """

    :param X: 输入特征矩阵
    :param params: 模型训练参数
    :return: 转换后模型的预测值
    """
    y_pred= sigmoid(np.dot(X, params["W"])+params["b"])
    # 基于分类阈值，对概率预测值类别转化
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred

def r2_score(y_test, y_pred):
    """
    :param y_test: 测试机标签
    :param y_pred: 测试集数据的预测值
    :return: R2值
    """
    y_ave = np.mean(y_test)
    # 总离差平方和，SST
    ss_tol = np.sum((y_test-y_ave)**2)
    # 残差平方和
    ss_res = np.sum((y_test-y_pred)**2)
    # R2
    r2 = 1-(ss_res/ss_tol)
    print("R2值为"+format(r2, ".3f"))
    return r2

def plotly_decision_boundary(x_train, y_train, params):
    """
    :param x_train: 训练数据
    :param y_train: 训练数据的标签
    :param params: 模型参数
    :return: 绘制Logit回归决策边界的可视化绘图
    """
    # 样本数量
    n = x_train.shape[0]
    x_label_cls1 = list()  # 类别1,x坐标
    y_label_cls1 = list()  # 类别1,y坐标
    x_label_cls2 = list()  # 类别2,x坐标
    y_label_cls2 = list()  # 类别1,y坐标
    for i in range(n):
        if y_train[i]==1:
            x_label_cls1.append(x_train[i][0])
            y_label_cls1.append(x_train[i][1])
        else:
            x_label_cls2.append(x_train[i][0])
            y_label_cls2.append(x_train[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_label_cls1, y_label_cls1, s=32, c="red")
    ax.scatter(x_label_cls2, y_label_cls2, s=32, c="green")

    x = np.arange(-1.5, 3, 0.1)
    print(params)
    y = (-params["b"]-params["W"][0]*x)/params["W"][1]
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("Y1")
    plt.show()






