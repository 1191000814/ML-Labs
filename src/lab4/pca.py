"""
PCA算法
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D

def init_data(n, k, m):
    """
    生成一个n维的样本, 其中k维的方差很大, 其他维的方差很小,便于转化成k维
    :param n: 原维度
    :param k: 将要转化的维度
    :param m: 样本数量
    :return: n维(高斯)分布的样本
    """
    print("生成一个", n, "维的样本, 数据多分布在其中的",k, "维中")
    mean = [random.randint(-10, 10) for _ in range(n)] # 随机生成均值
    print("均值向量是:", mean)
    cov = np.eye(n)
    cov_list = [1.0 for _ in range(k)] + [0.1 for _ in range(n - k)] # 方差由一部分1和另一部分0.1构成
    random.shuffle(cov_list)
    for i in range(n):
        cov[i][i] = cov_list[i]
    print("协方差矩阵是:", cov)
    x = np.random.multivariate_normal(mean, cov, m)
    return x


def pca(x, n, k):
    """
    PCA算法
    :param x: 样本
    :param n: 原维度
    :param k: 降维后的维度
    :return: 新坐标系, 重构后的样本
    """
    m = x.shape[0]
    mean = np.mean(x, axis=0)  # 求出样本的均值
    print("求得的均值向量是:", mean)
    for i in range(n):
        for j in range(m):
            x[j][i] -= mean[i] # 对每个样本去中心化
    print("去中心化后的样本:", x)
    cov = x.T.dot(x)
    eig = np.linalg.eig(cov)

    eig_value = eig[0] # 特征值
    eig_vector = eig[1] # 特征向量
    print("全部的特征值是: ", eig_value)
    print("全部的特征向量是: ", eig_vector)
    # 下面是为了选出前k个最大的特征值
    sorted_index = np.argsort(eig_value)
    print(sorted_index)
    w = np.delete(eig_vector, sorted_index[:n - k], axis=1)
    """
    max_eig_value = []
    max_eig_lab = []
    for i in range(n):
        if len(max_eig_value) < k: # 如果最大值列表还没满
            max_eig_lab.append(i) # 加入这个标签
            max_eig_value.append(eig_value[i]) # 加入这个值
        elif eig_value[i] > min(max_eig_value): # 如果这个特征值比最小的要大
            lab = max_eig_value.index(min(max_eig_value))
            max_eig_value.remove(min(max_eig_value)) # 移除最小值
            max_eig_value.append(eig_value[i]) # 加入这个值
            max_eig_lab.remove(lab) # 移除最小值标签
            max_eig_lab.append(i) # 加入这个标签
    """
    x1 = x.dot(w)
    x2 = x.dot(w).dot(w.T)
    print("降维后的特征向量: ", w)
    print("降维后的数据: ", x1)
    if x.shape[1] == 2:
        show(x, x2)
    elif x.shape[1] == 3:
        show3D(x, x2)
    x2[:, 0] += mean[0]
    x2[:, 1] += mean[1]
    print("重构后的数据: ", x2)
    return w, x2
    # 取出较大的k个特征值


def show3D(x, x1):
    """
    将样本集合m的数据在 3 维空间中显示出来
    :param x: 样本
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2])
    xx = np.arange(-10, 10, 0.5)  # x轴
    yy = np.arange(-10, 10, 0.5)  # y轴
    X, Y = np.meshgrid(xx, yy)  # 生成网格坐标
    # ax.plot_surface(X, Y, Z)
    plt.show()


def show(x, x1):
    """
    显示二维数据
    :param x 样本
    :return:
    """
    plt.scatter(x[:, 0], x[:, 1])
    plt.plot(x1[:, 0], x1[:, 1])
    plt.show()


if __name__ == '__main__':
    x = init_data(2, 1, 20)
    print(x)
    w, x1 = pca(x, 2, 1)