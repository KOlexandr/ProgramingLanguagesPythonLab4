__author__ = 'KOL'

from pylab import *
from scipy.interpolate import UnivariateSpline


# Setting up fake data with artificial noise
sample = 30
x = np.linspace(1, 10 * np.pi, sample)
y = np.cos(x) + np.log10(x) + np.random.random(sample) / 10

# Interpolating the data
f = UnivariateSpline(x, y, s=1)

# x.min and x.max are used to make sure we do not
# go beyond the boundaries of the data for the
# interpolation.
xInt = np.linspace(x.min(), x.max(), 1000)
yInt = f(xInt)

plot(x, y, 'o')
plot(x, y, 'g', label='Original')
plot(xInt, yInt, 'r', label='Interpolation')
plt.legend(loc='upper left', numpoints=1)
savefig("ex1.2.png")
show()