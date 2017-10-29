import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

fig= plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.set_xlim([0,5])
axes.set_ylim([0,5])

x = np.linspace(0,5)
#test how to use global variables.
#if you just want global value,then you don't need use global key word.
#but if you need to modify it,global key word cannot be missed.
a = 3
N = 2
lines = [axes.plot([], [])[0] for _ in range(N)]
lines[0],= axes.plot(x,x*0.1)
lines[1], = axes.plot(x,(x+0.1)*(x+0.1))

def init():    
    for line in lines:
        line.set_data([], [])
    return lines

def animate(i):
    global a
    a += 1
    print(a)
    lines[0].set_ydata(x*i)
    lines[1].set_ydata((x*i)*(x*i))
    #for line in enumerate(lines):
    #    line.set_ydata()
    return lines
#ani = 一定要写
ani = animation.FuncAnimation(fig,animate,np.linspace(0.1,10,100),interval = 50)

plt.show()
