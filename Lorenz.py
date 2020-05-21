import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

# rho = 28.0
rho = 14
sigma = 10.0
beta = 8.0 / 3.0


h = 0.01

def f(state):
    x, y, z = state  # Unpack the state vector
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])  # Derivatives

x = np.array([2.0, 2.0, 1.0])
t = np.arange(0.0, 20.0, 0.01)

# states = odeint(f, state0, t)
data = []
for i in t:

    k1 = f(x)
    k2 = f(x + h*(k1/2))
    k3 = f(x + h*(k2/2))
    k4 = f(x + h*k3)
    x_new = x + (h/6)*(k1 + 2*k2+ 2*k3 + k4)
    data.append(np.array((x_new)).reshape((1,3)))
    x = x_new

data = np.array([data]).reshape((-1,3))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(data[:, 0], data[:, 1], data[:, 2])
plt.draw()
plt.show()
