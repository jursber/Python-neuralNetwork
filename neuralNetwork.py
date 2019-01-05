# -*- coding: utf-8 -*-
"""
神经网络基本框架
初始化、训练、查询
"""
import numpy as np
from scipy.special import expit, logit
import os


class neuralNetwork:
    #初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #神经网络各层初始参数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #学习率
        self.lr = learningrate

        if self.init_check():
            #若存在权重文件，则使用
            self.wih = np.loadtxt(
                open("weight_data/wih.csv", "r"), delimiter=",", skiprows=0)
            self.who = np.loadtxt(
                open("weight_data/who.csv", "r"), delimiter=",", skiprows=0)
        else:
            #若不存在权重文件，初始化权重网络,mean=0,std=1/((连接数)^(1/2))
            self.wih = np.random.normal(0.0, pow(self.hnodes, -.5),
                                        (self.hnodes, self.inodes))
            self.who = np.random.normal(0.0, pow(self.onodes, -.5),
                                        (self.onodes, self.hnodes))

        #利用现有库创建Sigmoid函数
        self.activation_function = expit

    #验证是否存在已训练好的权重文件，文件是否有效
    def init_check(self):
        if os.path.exists('weight_data/wih.csv') and os.path.exists(
                'weight_data/who.csv'):
            if (np.loadtxt(
                    open("weight_data/wih.csv", "r"), delimiter=",",
                    skiprows=0).shape == (self.hnodes, self.inodes)
                    and np.loadtxt(
                        open("weight_data/who.csv", "r"),
                        delimiter=",",
                        skiprows=0).shape == (self.onodes, self.hnodes)):
                return True
        return False

    #训练神经网络
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #输出结果计算
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #反向传播误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)  #舍弃归一化因子后的近似误差

        #更新权重
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1 - final_outputs)),
            np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1 - hidden_outputs)),
            np.transpose(inputs))

    #神经网络结果计算
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        #隐藏层计算
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #输出层计算
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    #逆向查询，通过输出数组生成可解释的图像数组（并非原图）
    def back_query(self, target_list):
        #Sigmoid的反函数
        self.inverse_activation_function = logit
        #输出层
        final_outputs = np.array(target_list, ndmin=2).T
        final_inputs = self.inverse_activation_function(final_outputs)

        #隐藏层
        hidden_outputs = np.dot(self.who.T, final_inputs)
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        #输入层
        inputs = np.dot(self.wih.T, hidden_inputs)

        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        return inputs
