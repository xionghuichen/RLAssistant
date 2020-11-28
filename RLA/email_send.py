# coding:utf8
# python2.7 mailtest.py
'''
使用smtp和pop3 协议收发qq邮箱实验 
用户名和密码需要自己填写
'''

from smtplib import SMTP
from smtplib import SMTPRecipientsRefused
from poplib import POP3
from time import sleep
import sys

# 组合邮件格式
def send_error(sub, info, source_info, target_info):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("1.1.1.1", 80))
    ipaddr=s.getsockname()[0]
    s.close()
    sub = "ip %s, title %s " %(str(ipaddr), sub)
    origHeaders = ['From: {}'.format(source_info['from']),
                   'To: {}'.format(target_info['targer_email']),
                   'Subject: %s' %sub]
    origMsg = '\r\n\r\n'.join(['\r\n'.join(origHeaders),'\r\n'.join(info)])

    # 发送邮件部分
    sendSer = SMTP(source_info['smtpserver'])
    sendSer.set_debuglevel(1)
    print(sendSer.ehlo()[0])  # 服务器属性等
    sendSer.login(target_info['source_email'], target_info['source_passward'])  # qq邮箱需要验证
    try:
        errs = sendSer.sendmail(target_info['source_email'], target_info['targer_email'], origMsg)
    except SMTPRecipientsRefused:
        print('server refused....')
        sys.exit(1)
    sendSer.quit()
    assert len(errs) == 0, errs


    print('\n\n\nsend a mail ....OK!')
    sleep(10)  # 等待10秒
    print('Now get the mail .....\n\n\n')
