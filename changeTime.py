# -*- coding:utf8 -*-
import time

'''
功能：将timestamp类型的时间转化为年月日格式的时间
输入参数：timestamp类型的时间，类型为整型
输出参数：返回一个list，对应关系如下
          L[0]: 年
          L[1]: 月
          L[2]: 日
          L[3]: 时
          L[4]: 分
          L[5]: 秒
          L[6]: 星期（周日为0，周六为6）
'''
def ChangeTime(timestamp):
    names = ["Y","m","d","H","M","S",'w']
    tempTime = time.strftime("%Y-%m-%d-%H-%M-%S-%w",time.localtime(timestamp))
    T = tempTime.split('-')
    ret = dict()
    for i in range(len(T)):
        ret[names[i]] = int(T[i])
    return ret

# ChangeTime(time.time())