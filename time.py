#coding=utf-8
#time方法
import time
print(time.time())#返回当前的时间戳（1970以后）

#计算时间消耗，特别是对程序运行时间消耗
start = time.time()
for _ in range(100000000):
    pass
end = time.time()
print("循环运行时间:%.2f秒"%(end-start))
#output:循环运行时间:5.50秒
