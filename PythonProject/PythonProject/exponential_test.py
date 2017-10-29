import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.linspace(-2,2,1000)

#fig = plt.figure()

#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

#axes.plot(x,np.exp(x),lw= 2,color="red")

fig, ax = plt.subplots()

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0

ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0

#xx = np.linspace(-0.75, 1., 100)
ax.plot(x, np.exp(x),color="red")

ax.plot(x,x+1,color="green")

ax.plot(x,np.power(3,x),color="blue")

ax.plot(x,np.power(2,x),color="black")

#ax.plot(x,np.power(1,x))

plt.show()