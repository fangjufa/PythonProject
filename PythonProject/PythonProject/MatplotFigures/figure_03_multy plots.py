import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5,10)

fig,axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x, x**2, x, x**3)
axes[0].set_title("default axes ranges")

axes[1].plot(x, x**2, x, x**3)
axes[1].axis('tight')
axes[1].set_title("tight axes")

axes[2].plot(x, x**2, x, x**3)
axes[2].set_ylim([0, 60]) #设置自定义的y轴范围，而不是根据最上面通过linspace得到的数据，从0到60
axes[2].set_xlim([2, 5]) #设置自定义的x轴范围，从2到5.
axes[2].set_title("custom axes range");


plt.show()