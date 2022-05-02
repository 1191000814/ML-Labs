"""
混合高斯模型 GMM
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import math

class Gmm:
    """
    高斯混合模型
    """
    def __init__(self, k, n, m):
        self.k = k # 数量
        self.n = n # 维度
        self.m = m # 总体混合分布的全部样本数量
        self.cov = [0.2 * np.eye(n) for _ in range(k)] # 协方差矩阵
        self._mean = [] # 均值, 包含k个(n * 1)的array
        self._a = npr.random((k,))  # 随机生成混合模型的混合系数, 该数组之和要为1
        self._cluster = []
        self.x = [] # 一个array类型的list, 包含m个(n * 1)的array
        sum = np.sum(self._a)
        for i in range(k):
            self._a[i] /= sum
            self._mean.append(np.zeros((n,)) + i)
            # 初始均值为0, 2, 4 ... 尽量分散开
        print("成分系数: ", self._a)
        print("均值向量: ", self._mean)
        m0 = self.m
        for i in range(k): # # 每一个多维高斯分布,一共要生成k个混合高斯成分
            m1 = int(self._a[i] * m) # 这个成分的样本数
            if i < k - 1:
                num = m1 # 生成第k个多维高斯分布
                x0 = np.random.multivariate_normal(self._mean[i], self.cov[i], num)
                print(m1)
            else:
                num = m0 # 最后一个的数量等于总数减去其他数量
                x0 = np.random.multivariate_normal(self._mean[i], self.cov[i], num)
                print(m0)
            for j in range(num):
                self.x.append(x0[j])
            m0 -= m1


def k_means(gmm):
    """
    k-means算法
    :param gmm: 高斯混合模型
    :return: 聚类之后的均值, 分成的簇结果
    """
    k = gmm.k
    m = gmm.m
    n = gmm.n
    mean = [] # k个均值
    x = gmm.x # m个样本
    print("x =", len(x))
    print("m =", m)
    cluster = [[] for _ in range(k)] # 每个簇中的向量的索引
    for i in range(k): # 初始化k个均值向量
        mean.append(np.random.random((n, )) + i - 0.5)
    print("k-means 初始均值:", mean)
    t = 0 # 迭代次数
    while t < 100:
        cluster = [[] for _ in range(k)]  # 清空簇
        for i in range(m): # gmm中的每个样本
            dis = math.inf  # 该样本与均值的最小距离
            c = 0  # 属于哪个簇
            for j in range(k): # 每个均值向量
                if dis > np.linalg.norm(x[i] - mean[j]): # 计算欧氏距离
                    dis = np.linalg.norm(x[i] - mean[j]) # 取最小值
                    c = j # 改变所属的簇
            cluster[c].append(i)
        t += 1
        for i in range(k): # 更新第k个簇的均值向量
            if len(cluster[i]) == 0: # 空的簇无需考虑,防止除零错误
                continue
            sum = 0
            for j in cluster[i]: # j是第i个簇中的样本在全部样本中的序号
                sum += gmm.x[j]
            mean[i] = sum / len(cluster[i]) # 更新新的均值

    print("迭代次数: ", t)
    print("k-means算法分成的簇:", cluster)
    print("k-means算法计算的均值:", mean)
    return mean, cluster


def em(gmm):
    """
    EM 算法
    :param gmm: 高斯混合模型
    :return: 聚类之后的均值, 分成的簇结果
    """
    k = gmm.k
    m = gmm.m
    n = gmm.n
    x = gmm.x  # m个样本
    cov = [np.eye(n) for _ in range(k)]
    mean, cluster = k_means(gmm) # 用k-means算法求出em算法的初始值
    a = []  # 混合系数
    for i in range(k):
        a.append(len(cluster[i]) / m)
    print(a)
    t = 0

    while t < 50:
        gamma = [[0 for _ in range(k)] for _ in range(m)] # 后验概率矩阵
        # E步: 每一个样本对每一个簇的后验概率
        for i in range(m): # 第i个样本
            for j in range(k): # 第k个簇
                gamma[i][j] = app(gmm.x[i], mean, cov, a, j)

        # M步: 更新全部的系数: 均值, 协方差矩阵, 混合系数(一共k组)
        for j in range(k): # 第j个簇的参数
            p1 = np.zeros((n, )) # 均值的分子
            p2 = 0 # 均值的分母/协方差的分母/混合系数的分子
            p3 = np.zeros((n, n)) # 协方差的分子

            for i in range(m):
                p1 += (gamma[i][j] * x[i])
                p2 += gamma[i][j]
            mean[j] = p1 / p2 # 第j个簇的均值
            for i in range(m):
                p3 += (gamma[i][j] * (x[i] - mean[j]).reshape(n, 1) @ (x[i] - mean[j]).reshape(n, 1).T)
            cov[j] = p3 / p2 # 第j个簇的协方差,注意这个公式里的均值是已经更新的均值
            a[j] = p2 / m

        print(t)
        t += 1

    cluster = [[] for _ in range(k)]
    # 对样本重新分簇
    for i in range(m):
        dis = math.inf  # 该样本与均值的最小距离
        c = 0  # 属于哪个簇
        for j in range(k):  # 每个均值向量
            if dis > np.linalg.norm(x[i] - mean[j]):  # 计算欧氏距离
                dis = np.linalg.norm(x[i] - mean[j])  # 取最小值
                c = j  # 改变所属的簇
        cluster[c].append(i)
    print("EM算法分成的簇:", cluster)
    print("EM算法计算的均值:", mean)
    return mean, cluster


def gdf(x, mean0, cov0):
    """
    均值为mean0, 协方差为 cov0的高斯密度函数在 x点处的值
    :param x: 样本
    :param mean0: 簇的均值
    :param cov0: 簇的协方差
    :return: 概率密度
    """
    n = x.shape[0]
    dif = x - mean0
    return (1 / ((2 * math.pi) ** (n / 2) * math.sqrt(np.linalg.det(cov0)))) * np.exp(- 0.5 * dif.T @ np.linalg.pinv(cov0) @ dif)


def app(x, mean, cov, a, j):
    """
    已知高斯混合模型中所有参数(mean, cov, a),求第 i个样本 x属于第 j个簇的后验概率
    即西瓜书上的gamma(j, i)
    gmm中的 mean 和 cov是不可知

    :param x: 样本值 (n,)
    :param mean: 所有簇的均值 [(n,) * k]
    :param cov: 所有簇的协方差矩阵 [(n,n) * k]
    :param a: 系数矩阵(k,)
    :param j: 簇标号
    :return: 后验概率
    """
    k = len(cov)
    p1 = a[j] * gdf(x, mean[j], cov[j]) # 分子
    p2 = 0
    for l in range(k):
        p2 += a[l] * gdf(x, mean[l], cov[l])
    return p1 / p2


def show(gmm, mean, cluster):
    """
    展示图像, 只能展示 2维以下的图像
    :param gmm: 高斯混合模型
    :param mean: 均值
    :param cluster: 分类后的簇
    """
    k = len(cluster)
    __mean = gmm._mean # 原始均值

    for i in range(k): # 每个簇
        x0 = [] # 横坐标
        x1 = [] # 纵坐标
        for j in cluster[i]:
            x0.append(gmm.x[j][0])
            x1.append(gmm.x[j][1])
        plt.scatter(x0, x1)
    x0 = []
    x1 = []
    for i in range(k): # 每个平均值
        x0.append(mean[i][0])
        x1.append(mean[i][1])
    plt.scatter(x0, x1, marker="*", s=200, c="gold")
    plt.show()


def testUci():
    """
    测试Uci中的数据
    """
    data = np.genfromtxt("1.data", delimiter=",")
    print(data.shape)
    k = 3  # 混合成分数
    n = 4  # 维度数
    m = 150  # 样本数
    mean = [np.zeros((n, )) for _ in range(k)] # k-mean算法的初始值
    for i in range(data.shape[0]):
        if i <= 50:
            data[i][4] = 0
            mean[0] += (data[i][:4]).T
        elif (i > 50) & (i <= 100):
            data[i][4] = 1
            mean[1] += (data[i][:4]).T
        else:
            data[i][4] = 2
            mean[2] += (data[i][:4]).T
    for i in range(k):
        mean[i] /= 50
    print("初始mean值是:", mean)
    gmm = Gmm(3, 4, 150)
    for i in range(n):
        gmm.x[i] = data[i][:4]
    em(gmm)

if __name__ == '__main__':
    # gmm = Gmm(k=4, n=2, m=200)
    # print("混合分布的样本:", gmm.x)
    # mean, cluster = k_means(gmm)
    # mean, cluster = em(gmm)
    testUci()
    # show(gmm, mean, cluster)