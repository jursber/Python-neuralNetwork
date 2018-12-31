# -*- coding: utf-8 -*-
"""
其他验证：
识别自定义图片
查看反向查询的图像
"""
import scipy.misc
import numpy as np
import os
import neuralNetwork as nt
from PIL import Image

#图片缩放为指定像素
def imag_scale(input_path,maxw,maxh):
    imag_name=os.listdir(input_path)
    if not os.path.exists(input_path+'/scaled_imag/'):
        os.makedirs(input_path+'/scaled_img/') 
    count=0
    for name in imag_name:
        try:
            imag_path=os.path.join(input_path,name)
            print(imag_path)
            im=Image.open(imag_path)
            newim=Image.new('RGB',(maxw,maxh),'white')
            w,h=im.size
            scale=min(maxw/w,maxh/h)
                    
            out=im.resize((int(w*scale),int(h*scale)))
            newim.paste(out,(0,0))
            save_path=input_path+'/scaled_img/'+name
            newim.save(save_path)
            print(newim.size)
            count += 1
        except:
            pass
    print('共计处理图片',count,'张')
    
#图片读取为可供使用的数组
def imag_to_array(imag_path):
    imag_array=scipy.misc.imread(imag_path,flatten=True)
    imag_data=255-imag_array.reshape(784)
    imag_data=(imag_data/255*.99+.01)
    return imag_data


if __name__=='__main__':   
    #    imag_scale('test_img',28,28)
    #初始化神经网络
    input_nodes,hidden_nodes,output_nodes=784,150,10
    learning_rate=.15
    n=nt.neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
    #检验导入图像的识别效果
    imag_1=imag_to_array('test_img/scaled_img/8.png')
    print(np.argmax(n.query(imag_1)))
    
    #根据输出的数组，反向生成图像
    dis_output=np.full(10,0.01)
    dis_output[1]=0.99
    x=n.back_query(dis_output)
    import tools
    tools.target_image(x)
    
    

    