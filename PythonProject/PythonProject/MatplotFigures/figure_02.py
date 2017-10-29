'''
一张图表里显示两个曲线图，并加上图例。
图例默认显示的位置是左上角，
当然我们也可以选择图例显示的位置，只要像如下设置即可：
ax.legend(loc=0) # let matplotlib decide the optimal location
ax.legend(loc=1) # upper right corner
ax.legend(loc=2) # upper left corner
ax.legend(loc=3) # lower left corner
ax.legend(loc=4) # lower right corner
... 还有更多的选择可以用。
'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 10)

fig = plt.figure()

axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#axes.plot(x,x**2,label = "y=x**2")
#axes.plot(x,x**3,label = "y=x***2")  #x的n次方，表达式为x**n

#上面的图例表示得不够好，我们也可以用数学式的表达方式表示图例
#关于python中样式文本的写法，在文本的引号开头和结尾用$符号，
#在引号前面的r表示raw，因为\a是有另外的解释意义的。
#这里为了使得\alpha表示我们想要表示的文本意思，需要加上r这个字符。
axes.plot(x,x**2,label = r"$y = \alpha^2$")
axes.plot(x,x**3,label = r"$y = \alpha^3$")

axes.set_xlabel(r'$\alpha$')
axes.set_ylabel('$y$')

axes.legend()  #加上图例，图例就是上面label中的字符串

plt.show();
