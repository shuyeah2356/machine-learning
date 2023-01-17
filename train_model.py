import numpy as np
from train_utils import initialize_params, linear_model, logit_model, lasso_loss

# 线性回归模型的训练过程
def train_linear(X, y, lr=0.001, epoch=1000):
    """
    输入参数
        :param X: 输入变量矩阵
        :param y: 输入的标签矩阵
        :param lr: 学习率
        :param epoch: 总迭代次数
    :return:
        loss_his: 每次迭代后的均方误差
        params: 优化后的参数字典
        gradient: 优化后的梯度信息
    """
    loss_list = list()    # 用于记录训练的损失值
    params = dict()
    grads = dict()
    w, b = initialize_params(X.shape[1])    # 初始化模型参数
    for i in range(1, epoch):
        y_hat, loss, dw, db = linear_model(X, y, w, b)
        b += (-lr)*db
        w += (-lr)*dw
        loss_list.append(np.sum(loss)/X.shape[0])
        # if i%100 == 0:
        #     print("epoch {} , loss {}".format(i, format(np.sum(loss)/X.shape[0], ".3f")))
        params = {
            "w": w,
            "b": b
        }
        grads = {
            "dw": dw,
            "db": db
        }
    return loss_list, params, grads
# logit函数的训练过程
def train_logit(X, y, lr=0.01, epoch=1000):
    """
    输入参数：
        :param X: 输入特征矩阵
        :param y: 输出标签向量
        :param lr: 学习率
        :param epoch: 训练轮数
    :return:
        cost_list:损失列表
        params: 模型参数
        grads: 参数梯度
    """
    dw, db = None, None
    W, b = initialize_params(X.shape[1])    # 初始化模型参数
    cost_list = list()    # 初始化参数列表
    for i in range(epoch):
        a, cost, dw, db = logit_model(X, y, W, b)
        # 参数更新
        W = W-lr*dw
        b = b-lr*db
        if i % 100 == 0:
            cost_list.append(cost)
        if i % 100 ==0:
            print("epoch{}, cost{}".format(i, cost))
    # 保存参数
    params = {
        "W": W,
        "b": b
    }
    grads = {
        "dw": dw,
        "db": db
    }
    return cost_list, params, grads


# 定义lasso回归模型的训练过程
def train_lasso(X, y, lr, epoch):
    loss_his_list = list()
    w, b = initialize_params(X.shape[0])
    for i in range(1, epoch):
        y_hat, loss, dw, db = lasso_loss(X, y, w, b, 0.1)
        w += dw * (-lr)
        b += db * (-lr)
        loss_his_list.append(loss)
        if i % 50 == 0:
            print("epoch:{}, loss:{}".format(i, loss))
    params = {
        "W": w,
        "b": b
    }
    grads = {
        "dw": dw,
        "db": db
    }
    return loss_his_list, params, grads





