"""
图片降维
"""
import math

import matplotlib.pyplot as plt

from pca import pca
from PIL import Image
import numpy as np

COUNT = 400 # 图片数量
WIDTH = 50 # 图片的宽
LENGTH = 50 # 图片的长
DIMENSION = LENGTH * WIDTH # 图片原维度
REDUCED_DIMENSION = 100 # 降维之后的维度

def load_image():
    """
    加载图片
    """
    count = 61853
    mix_array = np.zeros((COUNT, WIDTH * LENGTH))
    for i in range(COUNT): # 读取m张图片
        path = "./50x50_grey_anime_face/data/" + str(count + i) + "_2019.jpg"
        img = Image.open(path)
        # img.show()
        img_array = np.array(img)
        if i < 9:
            plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap="gray")
        mix_array[i] = img_array.reshape((DIMENSION,))
    plt.show()
    return mix_array


def dimensionality_reduction(mix_array):
    """
    用pca给图片降维
    """
    w, x_refactor = pca(mix_array, DIMENSION, REDUCED_DIMENSION)
    x_refactor = x_refactor.real.astype(int) # 只取其中的实数部分
    psnr_array = np.zeros((COUNT, )) # 信噪比集合
    print("重构之后的数据:",x_refactor)
    for i in range(COUNT):
        img_array = x_refactor[i].reshape((WIDTH, LENGTH))
        if i < 9:
            plt.subplot(3, 3, i + 1)
        psnr_array[i] = psnr(mix_array[i], x_refactor[i])
        # 下面这行决定图像的显示方式
        plt.imshow(Image.fromarray(np.uint32(img_array)).convert("L"))
    plt.show()
    print("所有图片的平均信噪比是", psnr_array.mean())


def psnr(array1, array2):
    """
    :param array1: 原图像数组(已被压缩成一维)
    :param array2:
    :return:
    """
    mse = 0
    for i in range(DIMENSION):
        mse += (array1[i] - array2[i]) ** 2
    mse /= DIMENSION # 均方误差
    return 20 * math.log10(255 / math.sqrt(mse)) # 信噪比


if __name__ == '__main__':
    mix_array = load_image()
    print("原数据:", mix_array)
    dimensionality_reduction(mix_array)