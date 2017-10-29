'''
之前的那些图表都只是折线图，现在我们画一下散点图、柱状图、直方图
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.linspace(-0.75,1,100)

n = np.array([0,1,2,3,4,5])

fig,axes = plt.subplots(1,4,figsize=(12,3))

#散点图的y值是随机生成的，但是randn里的参数还不明白是做什么用的，是随机种子吗？
axes[0].scatter(x,x + 0.25*np.random.randn(len(x)))
axes[0].set_title("scatter")
#这里align属性是该数据柱是对齐x轴刻度的哪里，中间还是左边还是右边。
#alpha是设置数据柱的透明度的。
#这里面的属性除了前两个之外，设置柱状图的属性的命令可以随意替换顺序，或者不写，就用默认的样式。
axes[1].bar(n,n**2,alpha = 0.5,align="center")
axes[1].set_title("bar")

#填充两个图像之间的空白处，可以设置填充的颜色和透明度等。
axes[2].fill_between(x,x**2,x**3,color="green",alpha = 0.5)
axes[2].set_title("fill between")

#绘直方图
hist = np.random.randn(10000)
axes[3].hist(hist)
axes[3].set_title("histgram")

plt.show()