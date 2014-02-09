__author__ = 'KOL'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import SmoothBivariateSpline as SBS


# Defining a function
ripple = lambda x, y: np.sqrt(x ** 2 + y ** 2) + np.sin(x ** 2 + y ** 2)

# Generating sample that interpolation function will see
xy = np.random.random((300, 2))
x, y = xy[:, 0], xy[:, 1]
sample = ripple(xy[:, 0] * 2, xy[:, 1] * 2)

grid_x, grid_y = np.mgrid[0:1:300j, 0:1:300j]

# Interpolating data
fit = SBS(x * 2, y * 2, sample, s=0.01, kx=2, ky=2)
interp = fit(np.linspace(0, 2, 300), np.linspace(0, 2, 300))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.plot(x, y, sample, "go")
ax.plot_surface(grid_x, grid_y, interp)
ax.view_init(50, -30)
plt.savefig("ex1.4.png")
plt.show()