__author__ = 'KOL'

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# Defining a function
ripple = lambda x, y: np.sqrt(x ** 2 + y ** 2) + np.sin(x ** 2 + y ** 2)

# Generating gridded data. The complex number defines
# how many steps the grid data should have. Without the
# complex number mgrid would only create a grid data structure
# with 5 steps.
grid_x, grid_y = np.mgrid[0:1:300j, 0:1:300j]

# Generating sample that interpolation function will see
xy = np.random.random((300, 2))
sample = ripple(xy[:, 0] * 2, xy[:, 1] * 2)

# Interpolating data with a cubic
grid_z0 = griddata(xy, sample, (grid_x, grid_y), method='cubic')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.plot(xy[:, 0], xy[:, 1], sample, "go")
ax.plot_surface(grid_x, grid_y, grid_z0)
ax.view_init(50, -30)
plt.savefig("ex1.3.png")
plt.show()