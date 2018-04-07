import sys
import matplotlib.pyplot as plt

def exchange(f):
    f1 = open(f)
    line_y = []
    for i in f1:
        i = i[:len(i)-1]
        num = float(i)
        line_y.append(num)
    f1.close()
    return line_y

def add(y):
    y2 = []
    _y = 0
    for i,p in enumerate(y):
        _y += p
        y2.append(_y)
    return  y2

location = "/home/zyang/OPA_DIFF/logfile/reward"
y1 = exchange(location)
#y1 = add(y1)
'''
location = "/home/zyang/OPA_DIFF/r0_vb0/loss.txt"
y3 = exchange(location)
'''
#plt.subplot(311)
plt.plot(y1,label=' reward')
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
'''
plt.subplot(312)
plt.plot(y2,color='r',label='reward')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.legend()

plt.subplot(313)
plt.plot(y3,color='y',label='loss')
plt.xlabel('timestep')
plt.ylabel('loss')
plt.legend()
'''
plt.show()
