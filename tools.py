# -*- coding: utf-8 -*-
"""
"""
import matplotlib.pyplot as plt
import numpy as np
import random

#输出数字图形 
def target_image(n):
    with open('mnist_dataset/mnist_train.csv','r') as training_data_file:
        training_data_list=training_data_file.readlines()
    values=training_data_list[n].split(',')
    image_array=np.asfarray(values[1:]).reshape((28,28))
    plt.imshow(image_array,cmap='Greys',interpolation='None')
    print('当前数字：{}'.format(np.asfarray(values[0])))
    
n=random.randint(0,9999)
target_image(n)