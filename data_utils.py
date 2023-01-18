from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_classification
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 线性回归模型 导入数据集
def data_setting(data, target, sep_rate=0.8):
    """
    :param data: 数据集
    :param target: 标签
    :return:
    """
    X, y = shuffle(data, target, random_state = 13)
    # 按照8：2划分训练集和测试集
    offset = int(X.shape[0] * sep_rate)
    X_train, y_train = X[: offset], y[:offset]    # 训练集的数据和标签
    X_test, y_test = X[offset: ], y[offset: ]    # 测试集的数据和标签

    # 训练集/测试集转化为列向量的形式
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    print("X_train's shape:", X_train.shape)
    print("X_test's shape:", X_test.shape)
    print("y_train's shape:", y_train.shape)
    print("y_test's shape:", y_test.shape)

    return X_train, y_train, X_test, y_test


def list_line_plt(data_list, title_name=""):

    x_axis = list(range(len(data_list)))
    plt.plot(x_axis, data_list)
    plt.title(title_name)
    plt.show()

    # plt.plot()

# logit模型生成模拟二分类的数据集并进行可视化
def logit_data_setting():
    # 生成模拟二分类的数据集
    X, labels = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=2
    )

    # 随机数生成器
    rng=np.random.RandomState(6)
    # 对于生成的特征数据添加一组均匀分布的噪声
    X += 2*rng.uniform(size=X.shape)

    # 标签数类别
    unique_labels =set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        x_k = X[labels == k]
        plt.plot(x_k[:, 0], x_k[:, 1],"*", markerfacecolor=col,
                  markersize=14)
    plt.title("simulate binary data set")
    # plt.show()
    return X, labels

def lasso_data_setting():
    data = np.genfromtxt('lasso_data.txt', delimiter=",")
    x = data[:, 0: 100]
    y = data[:,100].reshape(-1, 1)

# LDA算法的数据测试
def lda_data_setting():
    # 导入Iris数据
    data = datasets.load_iris()
    X, y = data.data, data.target
    X, y = X[y != 2], y[y != 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    return X_train, X_test, y_train, y_test