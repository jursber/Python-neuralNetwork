# -*- coding: utf-8 -*-
"""
《Python神经网络教程》练习
"""
import numpy as np
from scipy.special import expit

class neuralNetwork:
    #初始化神经网络
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #神经网络各层初始参数
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        
        #学习率
        self.lr=learningrate
        
        #初始化权重网络,mean=0,std=1/((连接数)^(1/2))
        self.wih=np.random.normal(0.0,pow(self.hnodes,-.5),(self.hnodes,self.inodes))
        self.who=np.random.normal(0.0,pow(self.onodes,-.5),(self.onodes,self.hnodes))   

        #Sigmoid函数
        self.activation_function=expit
            
    #训练神经网络
    def train(self,inputs_list,targets_list):            
        inputs=np.array(inputs_list,ndmin=2).T
        targets=np.array(targets_list,ndmin=2).T
        
        #输出结果计算
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        
        #反向传播误差
        output_errors=targets-final_outputs
        hidden_errors=np.dot(self.who.T,output_errors) #舍弃归一化因子后的近似误差
        
        #更新权重
        self.who+=self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))
        self.wih+=self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))
        
    #神经网络结果计算
    def query(self,inputs_list):
        inputs=np.array(inputs_list,ndmin=2).T
        #隐藏层计算
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        
        #输出层计算
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        
        return final_outputs
    
    
