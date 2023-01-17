import numpy as np

# 线性回归模型
# 1、定义线性回归模型主体
def linear_model(X, y, w, b):
    """
    输入：
        X:输入变量矩阵
        y:数据标签矩阵
        w:权重参数矩阵
        b:偏置项
    return:
        y_hat:线性模型的预测值
        loss:均方误差
        dw:权重系数一阶偏导
        db：偏置项一阶偏导
    """
    # 样本量
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = np.dot(X, w)+b
    loss = np.abs(y_hat-y)
    dw = np.dot(X.T, (y_hat-y))/num_train
    db = np.sum(y_hat-y)/num_train
    return y_hat, loss, dw, db
# 2、初始化模型参数
def initialize_params(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

# sigmoid函数
def sigmoid(x):
    """

    :param x: 输入数组
    :return: sigmoid函数计算后的数组
    """
    z = 1/(1 + np.exp(-x))
    return z
# logitm回归模型的训练过程
def logit_model(X, y, W, b):
    """
    :param X:  输入特征矩阵
    :param y: 输出标签向量
    :param W: 权重系数
    :param b: 偏执参数
    :return:
    """
    # 训练的样本量
    num_train = X.shape[0]
    # 训练特征数量
    num_feature = X.shape[1]
    # 对数几率回归模型输出
    a = sigmoid(np.dot(X, W) + b)
    cost = -1/num_train * np.sum(y*np.log(a)+(1-y)*np.log(1-a))
    dw = np.dot(X.T, (a-y))/num_train
    db = np.sum(a-y)/num_train
    # 压缩数组维度
    cost = np.squeeze(cost)
    return a, cost, dw, db


# lasso回归模型定义一个符号函数
def sign_func(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


# lasso回归的损失函数
def lasso_loss(X, y, w, b, alpha):
    train_num = len(X.shape[0])
    feature_num = len(X.shape[1])
    y_hat = np.dot(X, w) + b
    loss = np.sum((y_hat-y)**2)/train_num + np.sum(alpha*abs(w))
    # 基于向量化符号函数的参数梯度计算
    vec_sign = np.vectorize(sign_func)  # 函数向量化
    dw = np.dot(X.T, (y_hat-y))/train_num + alpha*vec_sign(w)
    db = np.sum(y_hat-y)/train_num

    return y_hat, loss, dw, db

def ridge_loss(X, y, w, b, alpha):
    train_num = len(X.shape[0])
    feature_num = len(X.shape[1])
    y_hat = np.dot(X, w) + b
    loss = np.sum((y_hat - y) ** 2) / train_num + np.sum(alpha * np.square(w))

    dw = np.dot(X.T, (y_hat - y)) / train_num + alpha * 2*w
    db = np.sum(y_hat - y) / train_num

    return y_hat, loss, dw, db