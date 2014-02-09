__author__ = 'KOL'

from pylab import *
from scipy.interpolate import interp1d


# Setting up fake data
x = np.linspace(0, 10 * np.pi, 20)
y = np.cos(x)

# Interpolating data
fl = interp1d(x, y, kind='linear')
fq = interp1d(x, y, kind='quadratic')

# x.min and x.max are used to make sure we do not
# go beyond the boundaries of the data for the
# interpolation.
xInt = np.linspace(x.min(), x.max(), 1000)
yIntL = fl(xInt)
yIntQ = fq(xInt)

plot(x, y, 'o')
plot(xInt, yIntL, 'r', label="Linear")
plot(xInt, yIntQ, 'g', label="Quadratic")
plt.legend(loc='upper left', numpoints=1)
savefig("ex1.1.png")
show()