# -*- coding: utf-8 -*-
"""
微信提醒
"""

import itchat
import time
import threading
import time

#扫码登陆，保持
itchat.auto_login(True)

#是否发送消息
def message_toggle(arg):
    def outer(func):
        def wrapper(*karg,**kwargs):
            if arg:
                func(*karg,**kwargs)
            else:
                print(*karg,**kwargs)
        return wrapper
    return outer


#主动发送微信消息
@message_toggle(True)
def send_wechat_message(msg):
    users = itchat.search_friends(name='李宁')
    userName = users[0]['UserName']
    itchat.send(msg,userName)

#自动回复查询消息
def auto_wechat_reply(tid,rep):
    @itchat.msg_register(itchat.content.TEXT)
    def print_content(msg):
        myUserName = itchat.get_friends(update=True)[0]["UserName"]##获取自己的username
        time.sleep(1)
        if not msg['FromUserName'] == myUserName:###如果不是自己发的
            if msg.Content==tid:
                itchat.send_msg(rep,msg['FromUserName'])
    
    #loop，监听收到消息事件
    itchat.run()
    
    
#自动回复消息   
global real_time_progress
real_time_progress=None

def auto_reply(password): #微信输入password，自动回复real_time_progress
    global wcr_rep_flag,wcr_rep_Content
    flag=False
    def msg():
        itchat.auto_login(True)
#        time.sleep(10)
        global wcr_rep_flag,wcr_rep_Content
        while True:
            wcr_rep_Content=itchat.get_msg()
            if wcr_rep_Content[0] != None:
                if not len(wcr_rep_Content[0])==0:
                    wcr_rep_Content=wcr_rep_Content[0][0]['Content']
                    wcr_rep_flag=True
            time.sleep(.05)
    th=threading.Thread(target=msg)
    th.setDaemon(True)
    th.start()
    
    while True:
        if flag and wcr_rep_Content==password:
            global real_time_progress
            users = itchat.search_friends(name='李宁')
            userName = users[0]['UserName']
            itchat.send(str(real_time_progress),userName)
            wcr_rep_flag=False
            
            
