import numpy as np
import matplotlib
import matplotlib.pyplot as plt



x = np.linspace(0,5,10)

#������﷨��̫���
fig , axes = plt.subplots(1,2,figsize=(10,3))  #��������ͼ���һ������

axes[0].plot(x,x**2,x,x**3,lw = 2)  #lw means line width
axes[0].grid(True)  #������,��ʽ��Ĭ�ϵġ�

axes[1].plot(x,x**2,x,x**3,lw = 2)
axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5) #������ʽ

plt.show()