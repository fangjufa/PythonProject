import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100) #在0～5之间，生成100个数
#y = x ** 2
y = -x * np.log2(x)

fig = plt.figure()

'''
left, bottom, width, height (range 0 to 1)
这里设置的是图像显示的区域，是显示在窗口相对比例的地方。
比如[0.1  0.1  0.8  0.8]，假如窗口大小为100x100像素大小，
那么图像就显示在左下角起始点为(10,10)像素点，图像的长宽都为80像素。
'''
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes1.plot(x, y, 'r')

axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('title')

axes2 = fig.add_axes([0.2,0.5,0.35,0.35])
axes2.plot(y,x,'r')
axes2.set_xlabel('x')
axes2.set_ylabel('y')
axes2.set_title('insert title')


plt.show()