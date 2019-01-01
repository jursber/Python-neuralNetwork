# -*- coding: utf-8 -*-
"""
微信提醒
"""

import itchat
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
    