__author__ = 'KOL'

import numpy as np
import matplotlib.pyplot as mpl
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hy


def write_to_file(file_name, data_for_writing):
    """
    write coordinates of points into file for testing this data in another program, for example program from
    lab #6 from "Data mining" course
    """
    file = open(file_name, "w")
    x, y, z = data_for_writing[:, 0], data_for_writing[:, 1], data_for_writing[:, 2]
    for i in range(0, len(x)):
        file.write(str(x[i]) + "," + str(y[i]) + "," + str(z[i]) + "\n")
    file.close()


# Creating a cluster of clusters function
def clusters(number=20, c_number=5, c_size=10):
    # Note that the way the clusters are positioned is Gaussian randomness.
    #create random array (size c_number*2) with double values
    r_num = np.random.random((c_number, 2))
    #gets all values from first column and multiply they by number (20)
    rn = r_num[:, 0] * number
    #cast all numbers from rn to int
    rn = rn.astype(int)
    #find all indexes of numbers which less then 5 and set in this positions value 5
    rn[np.where(rn < 5)] = 5
    #find all indexes of numbers which bigger then number/2 (in our example: 10)
    #and set in this positions value number/2 (10)
    rn[np.where(rn > number / 2.)] = round(number / 2., 0)
    #gets all values from second column of r_num and multiply they by 2.9
    ra = r_num[:, 1] * 2.9
    #find all indexes of numbers which less then 1.5 and set they 1.5
    ra[np.where(ra < 1.5)] = 1.5
    #create new random array with values [0, number] and with size c_size*3
    cls = np.random.random((number, 3)) * c_size

    # Random multipliers for central point of cluster
    #create random array (c_number-1)*3
    rxyz = np.random.random((c_number - 1, 3))
    #iterate from 0 to c_number-1
    for i in range(c_number - 1):
        #create random array with size rn[i+1]*3
        tmp = np.random.random((rn[i + 1], 3))
        #create 3 vectors of coordinates of point (x,y,z) as sum of another random points created earlier
        x = tmp[:, 0] + (rxyz[i, 0] * c_size)
        y = tmp[:, 1] + (rxyz[i, 1] * c_size)
        z = tmp[:, 2] + (rxyz[i, 2] * c_size)
        #add all created coordinates to stack of points
        tmp = np.column_stack([x, y, z])
        #add coordinates to array (3 coordinates in one row represents one point)
        cls = np.vstack([cls, tmp])
    #return created random points
    return cls

# Generate a cluster of clusters and distance matrix.
#create random points
cls = clusters()
#count distance between all points
D = pdist(cls[:, 0:2])
#make square array with distances between points
D = squareform(D)

# Compute and plot first dendrogram.
fig = mpl.figure(figsize=(8, 8))
ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
#clissify data with hierarchical classification use method='complete' (d(u, v) = max(dist(u[i],v[j])))
#where d is distance
Y1 = hy.linkage(D, method='complete')
#find maximum value in third column of array Y1 and multiply in by 0.3
cutoff = 0.3 * np.max(Y1[:, 2])
#plots the hierarchical clustering as a dendrogram
Z1 = hy.dendrogram(Y1, orientation='right', color_threshold=cutoff)
#hide x axis
ax1.xaxis.set_visible(False)
#hide y axis
ax1.yaxis.set_visible(False)

# Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
#clissify data with hierarchical classification use method='average'
#d(u,v) = \\sum_{ij} \\frac{d(u[i], v[j])} {(|u|*|v|)}
Y2 = hy.linkage(D, method='average')
#find maximum value in third column of array Y1 and multiply in by 0.3
cutoff = 0.3 * np.max(Y2[:, 2])
#plots the hierarchical clustering as a dendrogram
Z2 = hy.dendrogram(Y2, color_threshold=cutoff)
#hide x axis
ax2.xaxis.set_visible(False)
#hide y axis
ax2.yaxis.set_visible(False)

# Plot distance matrix.
ax3 = fig.add_axes([0.3, 0.1, 0.6, 0.6])
#gets from created dendrograms arrays with key 'leaves'
idx1 = Z1['leaves']
idx2 = Z2['leaves']
#and gets from distance matrix rows and columns with this indexes
D = D[idx1, :]
D = D[:, idx2]
#plot new distance matrix as image with different colors for different points
ax3.matshow(D, aspect='auto', origin='lower', cmap=mpl.cm.YlGnBu)
#hide x axis
ax3.xaxis.set_visible(False)
#hide y axis
ax3.yaxis.set_visible(False)

# Plot colorbar.
#save create figure with dendrograms and image which represent matrix of distances into pdf file
fig.savefig('cluster_hy_f01.pdf', bbox='tight')


# Same imports and cluster function from the previous example
# follow through here.
# Here we define a function to collect the coordinates of
# each point of the different clusters.
def group(data, index):
    number = np.unique(index)
    groups = []
    for i in number:
        groups.append(data[index == i])
    return groups

# Creating a cluster of clusters
cls = clusters()
write_to_file("data.txt", cls)
# Calculating the linkage matrix
Y = hy.linkage(cls[:, 0:2], method='complete')

# Here we use the fcluster function to pull out a
# collection of flat clusters from the hierarchical
# data structure. Note that we are using the same
# cutoff value as in the previous example for the dendrogram
# using the 'complete' method.
cutoff = 0.3 * np.max(Y[:, 2])
index = hy.fcluster(Y, cutoff, 'distance')

# Using the group function, we group points into their
# respective clusters.
groups = group(cls, index)

# Plotting clusters
fig = mpl.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
#array with colors for points on figure
colors = ['r', 'c', 'b', 'g', 'orange', 'k', 'y', 'gray']
#itarate by all groups of points (clusters)
for i, g in enumerate(groups):
    i = np.mod(i, len(colors))
    #plot all points from current cluster (first column - x-coordinate, second column - y-coordinate)
    ax.scatter(g[:, 0], g[:, 1], c=colors[i], edgecolor='none', s=50)
    #hide x axis
    ax.xaxis.set_visible(False)
    #hide y axis
    ax.yaxis.set_visible(False)
#save figure with points as pdf file
fig.savefig('cluster_hy_f02.pdf', bbox='tight')