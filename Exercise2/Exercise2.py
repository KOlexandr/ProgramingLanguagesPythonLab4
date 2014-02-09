__author__ = 'KOL'

from pylab import *
from scipy.cluster import vq


def write_to_file(file_name, data_for_writing):
    file = open(file_name, "w")
    x, y = data_for_writing[:, 0], data_for_writing[:, 1]
    for i in range(0, len(x)):
        file.write(str(x[i]) + "\t" + str(y[i]) + "\n")
    file.close()

# Creating data
c1 = np.random.random((100, 2)) + 5
#export c1 into file for testing in matlab
write_to_file("c1.txt", c1)
c2 = np.random.random((30, 2)) - 5
#export c2 into file for testing in matlab
write_to_file("c2.txt", c2)
c3 = np.random.random((50, 2))
#export c3 into file for testing in matlab
write_to_file("c3.txt", c3)

# Pooling all the data into one 180 x 2 array
data = np.vstack([c1, c2, c3])

# Calculating the cluster centroids and variance
# from kmeans
centroids, variance = vq.kmeans(data, 3)

# The identified variable contains the information
# we need to separate the points in clusters
# based on the vq function.
identified, distance = vq.vq(data, centroids)

# Retrieving coordinates for points in each vq
# identified core
vqc1 = data[identified == 0]
vqc2 = data[identified == 1]
vqc3 = data[identified == 2]

plot(vqc1[:, 0], vqc1[:, 1], 'ro', label="Class 1")
plot(vqc2[:, 0], vqc2[:, 1], 'go', label="Class 2")
plot(vqc3[:, 0], vqc3[:, 1], 'bo', label="Class 3")
plt.legend(loc='upper left', numpoints=1)
savefig("ex2.png")
show()