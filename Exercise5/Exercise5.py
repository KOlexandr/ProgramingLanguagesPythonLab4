__author__ = 'KOL'

from numpy import *
from matplotlib import cm
import matplotlib.pyplot as plt
from numpy.core.multiarray import zeros


def seidel_method(x, ax, ay, cx, cy, b, d, nx=31, ny=31, eps=1e-5):
    """
    The Gauss–Seidel method is an iterative technique for solving a square system of n linear equations with unknown x:
    A*x = b.
    """
    error = 1
    k_max = 500
    k = 0
    while error >= eps and k < k_max:
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                x_tmp = -d[i, j]
                x_tmp = x_tmp + ax[i, j]*x[i-1, j]
                x_tmp = x_tmp + cx[i, j]*x[i+1, j]
                x_tmp = x_tmp + ay[i, j]*x[i, j-1]
                x_tmp = x_tmp + cy[i, j]*x[i, j+1]
                x_tmp = x_tmp/b[i, j]
                dev = abs((x_tmp - x[i, j])/x_tmp + 1.0e-10)
                if dev > error:
                    error = dev
                x[i, j] = x_tmp
        k += 1
    return x


def main():
    """
    deflection of the lamina stretched over a frame under weight
    elliptic equation of second order with partial derivatives
    """
    #number of x points
    nx = 31
    #number of y points
    ny = 31
    #length of frame by x axis
    len_x = 1
    #length of frame by y axis
    len_y = 1
    #step by x
    hx = len_x/nx
    #step by y
    hy = len_y/ny

    #solution matrix
    z = zeros((nx, ny))
    #sets boundary conditions of the second order (we considers value of derivative on the left boundary of frame)
    z[:, 0] = 1

    #matrix of initial data (using method of cross)
    ax = tile(1/(hx**2), (nx, ny))
    cx = tile(1/(hx**2), (nx, ny))
    ay = tile(1/(hy**2), (nx, ny))
    cy = tile(1/(hy**2), (nx, ny))
    b = tile(2/(hx**2) + 2/(hy**2), (nx, ny))

    #right part of equation
    #we can change value in right part of equation and, as result, get new graphics
    d = tile(5, (nx, ny))
    #we can set some value as anomaly in function of right part of equation
    d[10, 17] -= 900

    #using seidel method
    z = seidel_method(z, ax, ay, cx, cy, b, d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #initialize x and y coordinates for plotting solution of equation
    #nx and ny may be different and we can use size of frame and not size of initial matrix
    x = zeros(shape=(nx, ny))
    x[:] = arange(0, nx)
    y = zeros(shape=(nx, ny))
    y[:] = arange(0, ny)
    y = y.transpose()

    #plots surface
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.view_init(50, -30)
    #saves result into png file
    plt.savefig("ex5.png")
    #shows result on screen at the moment
    plt.show()

main()