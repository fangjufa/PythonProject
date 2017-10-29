import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import time

from mpl_toolkits.mplot3d.axes3d import Axes3D


nx = np.linspace(0,1,3)
ny = np.linspace(0,1,4)

nnx,nny = np.meshgrid(nx,ny)

print(nnx)
print(nny)

fig = plt.figure(figsize=(14,6))

alpha = 0.7
#phi_ext = 2 * np.pi * 0.5 

def flux_qubit_potential(phi_m, phi_p):
    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(np.pi - 2*phi_p)

phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T

# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')

p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)

# surface_plot with color grading and color bar
ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)

plt.show()

#plt.canvas.draw()
