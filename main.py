from data_utils import data_setting, list_line_plt, logit_data_setting
from train_utils import linear_model, initialize_params
from predict_utils import predict, r2_score, predict_logit, plotly_decision_boundary
from sklearn.datasets import load_diabetes
from train_model import train_linear, train_logit
from sklearn.metrics import classification_report
from sklearn import linear_model


diabetes = load_diabetes()
# 获取数据和标签
data, target = diabetes.data, diabetes.target
set_rate = 0.8
X_train, y_train, X_test, y_test = data_setting(data, target, set_rate)
# -------------------------------------------------------------------
# 线性回归模型
# 训练模型
loss_list, params, grads = train_linear(X_train, y_train, 0.2, 10000)
# 绘制训练过程的损失函数曲线
list_line_plt(loss_list, "train_loss")
# 预测结果
prediction = predict(X_test, params)
# 回归模型结果的R2系数
r2_value = r2_score(y_test, prediction)

# ---------------------------------------------------------------------
# logit回归模型
# 生成二分类数据并查看数据
data_binary, label_data = logit_data_setting()
# 划分数据集
x_train, y_train, x_test, y_test = data_setting(data_binary, label_data, 0.8)
# 训练logit回归模型
cost, params, gradients = train_logit(x_train, y_train)
# 输出逻辑回归模型的预测结果
y_prediction = predict_logit(x_test, params)
# 模型评估
res = classification_report(y_test, y_prediction)
# 绘图-逻辑回归决策边界
plotly_decision_boundary(x_train, y_train, params)

# ------------------------------------------------------------------------
# LASSO(the least absolute shrinkage and selection operator)最小绝对收缩和选择算子
# lasso回归模型训练
