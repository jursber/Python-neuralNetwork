# -*- coding: utf-8 -*-
"""
神经网络训练、测试
"""

#导入神经网络类
import neuralNetwork as nt
import numpy as np
import time
import os
from multiprocessing import Pool
import scipy.ndimage
import weChatReminder as wcr
import threading


#将原数据图像旋转±10度，生成新的训练集，丰富训练数据
def rotate_data():
    start_time = time.time()
    #旋转函数
    plus10=lambda x:scipy.ndimage.interpolation.rotate(x.reshape(28,28),10,cval=.01,reshape=False).reshape(784)
    minus10=lambda x:scipy.ndimage.interpolation.rotate(x.reshape(28,28),-10,cval=.01,reshape=False).reshape(784)

    with open('mnist_dataset/mnist_train.csv', 'r') as training_data_file:
        training_data_list = training_data_file.readlines()

    #旋转后的目标值和训练值
    rotate_target_data = []
    rotate_train_data = []
    for record in training_data_list:
        rotate_target_data.append(np.asfarray(record.split(',')[0]))
        rotate_train_data.append(np.asfarray(record.split(',')[1:]))
    train_data_plus10 = np.array(list(map(plus10, rotate_train_data)))
    train_data_minus10 = np.array(list(map(minus10, rotate_train_data)))
    rotate_target_data = np.array(rotate_target_data)

    #合并数据
    csv_plus10 = np.c_[rotate_target_data, train_data_plus10]
    csv_minus10 = np.c_[rotate_target_data, train_data_minus10]

    #存入文件
    np.savetxt('mnist_dataset/new_train_plus10.csv', csv_plus10, delimiter=',')
    np.savetxt(
        'mnist_dataset/new_train_minus10.csv', csv_minus10, delimiter=',')

    wcr.send_wechat_message('训练集生成完毕，总计用时:{}s'.format(
        round(time.time() - start_time, 0)))

#读取训练文件
def read_csv(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return data

#训练神经网络
def training_neuralNework(training_data_list, epochs=1):
    #记录程序开始时间
    start_time = time.time()

    #删除原训练数据
    try:
        os.remove("weight_data/who.csv")
        os.remove("weight_data/wih.csv")
    except IOError:
        pass

    train_count = 0
    for e in range(epochs):  #开始训练
        print('第{}次循环'.format(e + 1))
        data_count = 1
        for record in training_data_list:
            all_values = record.split(',')
            #归一化输入，0~255 int变成0.01~1 float
            inputs = np.asfarray(all_values[1:]) / 255 * .99 + .01
            #归一化target，0~9变成数组
            targets = np.zeros(output_nodes) + 0.01
            targets[int(float(all_values[0]))] = 0.99
            #开始训练
            n.train(inputs, targets)
            train_count += 1
            #print(n.wih[:2,::560])
            #print(n.who[:2,::10])
        data_count += 1
        wcr.send_wechat_message('第{}次循环（训练）完成！，总计用时{}s'
            .format(e + 1, round(time.time() - start_time, 0)))

    #训练数据存入文件
    np.savetxt('weight_data/wih.csv', n.wih, delimiter=',')
    np.savetxt('weight_data/who.csv', n.who, delimiter=',')
    #记录程序运行时长
    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    wcr.send_wechat_message('训练完成，训练数据{}行，循环{}次，总计用时{:02d}:{:02d}:{:02d}'
                            .format(train_count / epochs, train_count, h, m, s))

#测试神经网络
def test_neuralNework():
    with open('mnist_dataset/mnist_test.csv', 'r') as test_data_file:
        test_data_list = test_data_file.readlines()
    #对识别效果计分
    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        #处理输入
        inputs = np.asfarray(all_values[1:]) / 255 * .99 + .01
        #通过神经网络输出
        outputs = n.query(inputs)
        #记录正确率结果的数量
        label = np.argmax(outputs)
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    wcr.send_wechat_message('{}个输出结果中，正确的个数为{}，正确率：{:.2%}'.format(
        len(scorecard), scorecard.count(1),
        scorecard.count(1) / len(scorecard)))

if __name__ == '__main__':
    wcr.send_wechat_message('Python神经网络程序开始运行！')
    #设置每层节点 & 学习率
    input_nodes, hidden_nodes, output_nodes = 784, 150, 10
    learning_rate = .15
    #创建神经网络对象
    n = nt.neuralNetwork(input_nodes, hidden_nodes, output_nodes,
                         learning_rate)

    rotate_data() #扩充训练集

    #多线程读取文件
    read_start_time = time.time()
    training_data_path = [
        'mnist_dataset/mnist_train.csv', 'mnist_dataset/new_train_minus10.csv',
        'mnist_dataset/new_train_plus10.csv'
    ]  # 全部训练集
    training_data_list = []  #全部训练集存入数组
    pool = Pool(processes=3)
    for f in training_data_path:
        training_data_list.extend(pool.apply_async(read_csv, (f, )).get())
    pool.close()
    pool.join()
    print('文件读取完成，读取用时：{}s'.format(int(time.time() - read_start_time)))

    #开始训练
    training_neuralNework(training_data_list,epochs=5)  
    #测试训练效果
    test_neuralNework()  #测试训练效果

#    itchat.logout()
