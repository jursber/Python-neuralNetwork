# -*- coding: utf-8 -*-
"""
神经网络训练、测试
"""

#导入神经网络类
import neuralNetwork as nt
import numpy as np
import time
import os

#训练神经网络
def training_neuralNework():
    #记录程序开始时间
    start_time = time.time()
    #删除原训练数据
    try:
        os.remove("weight_data/who.csv")
        os.remove("weight_data/wih.csv")
    except:pass
    #读取文件
    training_data_file=open('mnist_dataset/mnist_train.csv','r') 
    training_data_list=training_data_file.readlines()
    training_data_file.close()
    
    #开始训练
    train_count=0
    epochs=6 #训练集应用次数，世代
    for e in range(epochs):
        print('第{}次循环'.format(e+1))
        for record in training_data_list:
            all_values=record.split(',')
            #归一化输入，0~255 int变成0.01~1 float
            inputs=np.asfarray(all_values[1:])/255*.99+.01
            #归一化target，0~9变成数组
            targets=np.zeros(output_nodes)+0.01
            targets[int(all_values[0])]=0.99
            #开始训练
            n.train(inputs,targets) 
            train_count+=1
        #print(n.wih[:2,::560])
        #print(n.who[:2,::10])
        
    #训练数据存入文件
    np.savetxt('weight_data/wih.csv',n.wih,delimiter=',')
    np.savetxt('weight_data/who.csv',n.who,delimiter=',')
    #记录程序运行时长   
    m, s = divmod(int(time.time()-start_time), 60)
    h, m = divmod(m, 60)
    
    print('训练完成，训练数据{}行，循环{}次，用时{:02d}:{:02d}:{:02d}'
          .format(train_count/epochs,train_count,h,m,s))

#测试神经网络   
def test_neuralNework():     
    with open('mnist_dataset/mnist_test.csv','r') as test_data_file:
        test_data_list=test_data_file.readlines()
    
    #对识别效果计分
    scorecard=[]
    for record in test_data_list:
        all_values=record.split(',')
        correct_label=int(all_values[0])
        #处理输入
        inputs=np.asfarray(all_values[1:])/255*.99+.01
        #通过神经网络输出
        outputs=n.query(inputs)
        #记录正确率
        label=np.argmax(outputs)
        if label==correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    print('{}个输出结果中，正确的个数为{}，正确率：{:.2%}'
          .format(len(scorecard),scorecard.count(1),scorecard.count(1)/len(scorecard)))
    
    
if __name__=='__main__':
    #设置每层节点
    input_nodes,hidden_nodes,output_nodes=784,200,10
    #设置学习率
    learning_rate=.2
    #创建神经网络对象
    n=nt.neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
    
    #执行程序
    training_neuralNework()
    test_neuralNework()
























