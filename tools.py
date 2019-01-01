# -*- coding: utf-8 -*-
"""
实用工具，用于测试
"""
import matplotlib.pyplot as plt
import numpy as np
import random

#输出数字图形 
def target_image(values):
    image_array=np.asfarray(values).reshape((28,28))
    plt.imshow(image_array,cmap='Greys',interpolation='None')
    print('当前数字：{}'.format(np.asfarray(values[0])))


if __name__=='__main__':
    with open('mnist_dataset/mnist_train.csv','r') as training_data_file:
        training_data_list=training_data_file.readlines()
    n=random.randint(0,9999)
    values=training_data_list[n].split(',')[1:]
    target_image(values)