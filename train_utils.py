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
        db:偏置项一阶偏导
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

# ridge回归损失函数
def ridge_loss(X, y, w, b, alpha):
    train_num = len(X.shape[0])
    feature_num = len(X.shape[1])
    y_hat = np.dot(X, w) + b
    loss = np.sum((y_hat - y) ** 2) / train_num + np.sum(alpha * np.square(w))

    dw = np.dot(X.T, (y_hat - y)) / train_num + alpha * 2*w
    db = np.sum(y_hat - y) / train_num

    return y_hat, loss, dw, db

# LDA算法实现
"""
线性判别分析,是一种有监督的降维技术
两组数据投影到同一条直线上,优化策略是类内距离尽可能小,类间距离尽可能大
LDA实现步骤:
1、对训练集按照类别进行分组
2、分别计算每组样本的协方差
3、计算类间散度矩阵Sw
4、计算两类样本的均值差μ0-μ1
5、对类间散度矩阵Sw进行奇异值分解,并求其逆
6、根据Sw-1(μ0-μ1)得到W
7、最后计算投影后的数据点Y=WX
"""
class LDA:
    def __init__(self):
        # 初始化权重矩阵
        self.w = None

    # 协方差矩阵
    def calc_cov(self, X, Y=None):
        m = X.shape[0]
        # 数据标准化
        X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        if Y == None:
            Y = X
        else:
            (Y-np.mean(Y, axis=0))/np.std(Y, axis=0)
        return 1/m * np.matmul(X.T, Y)    # matmul计算协方差矩阵

    # 数据投影方法
    def project(self, X, y):
        # LDA拟合获取模型权重
        self.fit(X, y)
        # 数据投影
        X_projection = X.dot(self.w)
        return X_projection

    # LDA拟合方法
    def fit(self, X, y):
        # (1)按类分组
        X0 = X[y == 0]
        X1 = X[y == 1]
        # (2)分别计算两个类别数据自变量的协方差矩阵
        sigma0 = self.calc_cov(X0)
        sigma1 = self.calc_cov(X1)
        # (3)计算类内散度矩阵
        Sw = sigma0 + sigma1 
        # (4)分别计算两类数据自变量的均值、差
        u0, u1 = np.mean(X0, axis=0), np.mean(X1, axis=0)
        mean_diff = np.atleast_1d(u0 - u1)
        # (5)对类内三段矩阵进行奇异值分解
        U, S, V = np.linalg.svd(Sw)
        # (6)计算类内散度矩阵的逆
        Sw_ = np.dot(np.dot(V.T, np.linalg.pinv(np.diag(S))), U.T)
        # (7)计算w
        self.w = Sw_.dot(mean_diff)

    # LDA分类预测
    def predict(self, X):
        # 初始化预测结果为空列表
        y_pred = []
        # 遍历待预测样本
        for x_i in X:
            # 预测模型
            h = x_i.dot(self.w)
            y = 1*(h<0)
            y_pred.append(y)
        return y_pred

# k近邻算法KNN
"""
k近邻算法三个要素:
K值选择
距离度量方式
分类规则

k近邻算法实现相似推荐系统:
1、基于商品的推荐方法,为目标用户推荐一些他有购买偏好的商品的类似商品；
2、基于用户的推荐方法,用k近邻算法找到与目标用户喜好类似的用户,然后根据这些用户的喜好来向目标用户推荐
"""
