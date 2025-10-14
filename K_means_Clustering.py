# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs

# Creating/Importing Dataset
X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)
# make_blobs(n_samples=500, n_features=2, centers=3): Generates 500 data points in a 2D space, grouped into 3 clusters

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1]) # Plots the dataset in 2D, showing all the points
plt.show()

# Initializing Random Centroids
k = 3

# clusters is a dictionary where each key is the cluster index (0, 1, 2) and the value is another dictionary containing the centroid and an empty list for points
clusters = {} # Dictionary to hold the clusters
# Random Seeds : which means that every time you run the code, you will get the same random numbers
np.random.seed(23) # np.random.seed(23): Ensures reproducibility by fixing the random seed.

# The for loop initializes k random centroids, with values between -2 and 2, for a 2D dataset
for idx in range(k):
    center = 2*(2*np.random.random((X.shape[1],))-1) # X.shape[1] gives the number of features (2 in this case)
    # center = 2*(2*np.random.random((X.shape[1],))-1): Generates a random centroid in 2D space, with each coordinate ranging from -2 to 2

    points = []  # List to hold points assigned to this cluster
    cluster = {
        'center' : center,
        'points' : []
    }
    clusters[idx] = cluster 
clusters

# Plotting Random Initialized Center with Data Points
plt.scatter(X[:,0],X[:,1]) # Plots the dataset in 2D, showing all the points
plt.grid(True)
for i in clusters:
    center = clusters[i]['center'] # Accesses the centroid of each cluster
    plt.scatter(center[0],center[1],marker = '*',c = 'red') # Plots the cluster center as a red star (* marker).
plt.show()

# Defining Eculidean Distance Function
def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

# Creating Assign and Update Functions
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]): # Loops through each data point in X
        dist = [] # Initializes an empty list to store distances from the point to each cluster center

        curr_x = X[idx] # Gets the current data point

        for i in range(k):
            dis = distance(curr_x,clusters[i]['center']) # Calculates the distance from the current point to the i-th cluster center
            dist.append(dis) # Appends the calculated distance to the list dist.
            curr_cluster = np.argmin(dist) # Finds the index of the closest cluster by selecting the minimum distance.
            clusters[curr_cluster]['points'].append(curr_x) # Assigns the current data point to the closest cluster by appending it to the points list of that cluster.
    return clusters

def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points']) # Converts the list of points in the i-th cluster to a NumPy array for easier manipulation
        if points.shape[0] > 0: # Checks if the cluster has any points assigned to it
            new_center = points.mean(axis = 0) # Calculates the new centroid by taking the mean of the points in the cluster
            clusters[i]['center'] = new_center # Updates the cluster's center with the newly calculated centroid

            clusters[i]['points'] = [] # Resets the points list for the next iteration
    return clusters

# Predicting the Cluster for the Data Points
def pred_Clusters(X, clusters):
    pred = [] # Initializes an empty list to store the predicted cluster indices for each data point
    for i in range(X.shape[0]): # Loops through each data point in X
        dist = [] # Initializes an empty list to store distances from the point to each cluster center
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center'])) # Calculates the distance from the current point to the j-th cluster center and appends it to dist
            pred.append(np.argmin(dist)) # Appends the index of the closest cluster (the one with the minimum distance) to pred.
    return pred

# Assigning, Updating and Predicting the Cluster Centers
clusters = assign_clusters(X,clusters) # Assigns data points to the nearest centroids
clusters = update_clusters(X,clusters) # Recalculates the centroids
pred = pred_Clusters(X,clusters) # Predicts the final clusters for all data points

# Plotting Data Points with Predicted Cluster Centers
plt.scatter(X[:,0],X[:,1],c = pred)
for i in clusters:
    center = clusters[i]['center'] # Retrieves the center (centroid) of the current cluster
    plt.scatter(center[0],center[1],marker = '^',c = 'red') # Plots the cluster center as a red triangle (^ marker)
plt.show()