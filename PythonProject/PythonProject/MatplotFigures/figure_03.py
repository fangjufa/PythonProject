#这个代码文件得出一个结论，编码规格很重要。

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5,10)

fig = plt.figure()

axes = plt.subplots(1,2,figsize=(10,3))

#生成第一个曲线图
#axes[0].plot(x, x**2, x, x**3, lw=2)
axes[0].plot(x,x**2,label = r"$y = \alpha^2$")
axes[0].plot(x,x**3,label = r"$y = \alpha^3$")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].grid(True)

#生成第二个曲线图

axes[1].plot(x,x**2,x,x**3,lw = 2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

plt.show()