'''
设置双y轴，比如x轴是长度，一个y轴是面积，另一个y轴是体积
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.linspace(0,5,10)

fig,axes = plt.subplots()

axes.plot(x,x**2,lw = 2,color = "blue")
axes.set_ylabel(r"area($x^2$)",fontsize = 18,color = "blue")

#下面这个for循环，是设置y轴上每个刻度的颜色的。
#如果不设置的话，那它就会显示默认的黑色。
for label in axes.get_yticklabels():
    label.set_color("blue")

axes2 = axes.twinx()  #再创建一个轴,twinx是再创建一个y轴，twiny是再创建一个x轴。
axes2.plot(x,x**3,lw = 2,color = "red")
axes2.set_ylabel(r"Volume ($x^3$)",fontsize = 18,color="red")
for label in axes2.get_yticklabels():
    label.set_color("red")

plt.show()
    
    

